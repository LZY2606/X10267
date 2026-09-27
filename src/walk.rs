use std::cmp::Ordering;
use std::collections::HashSet;
use std::error::Error as StdError;
use std::ffi::OsString;
use std::fmt;
use std::fs::{self, DirEntry, Metadata, ReadDir};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use crate::{glob_match, validate, Error as PatternError};

static NEXT_INVOCATION_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PermissionDecision {
    Continue,
    Return,
}

/// Controls how filesystem permission errors are reported by directory walks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PermissionErrorPolicy {
    /// Record the directory as pruned and continue with the rest of the walk.
    Skip,
    /// Return the first permission error. No partially collected result is exposed.
    #[default]
    Return,
}

/// Options used by the public filesystem traversal entry points.
#[derive(Clone, Debug)]
pub struct WalkOptions {
    /// Match ASCII path names case-insensitively when false.
    pub case_sensitive: bool,
    /// Ignore dotfiles and directories containing a dot segment when true.
    pub ignore_hidden: bool,
    /// Follow symbolic links into directories when true.
    pub follow_symlinks: bool,
    /// Selects whether permission errors stop the walk or are skipped.
    pub permission_errors: PermissionErrorPolicy,
}

impl Default for WalkOptions {
    fn default() -> Self {
        Self {
            case_sensitive: true,
            ignore_hidden: false,
            follow_symlinks: false,
            permission_errors: PermissionErrorPolicy::default(),
        }
    }
}

impl WalkOptions {
    /// Creates options matching the existing matcher's case-sensitive semantics.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn case_sensitive(mut self, case_sensitive: bool) -> Self {
        self.case_sensitive = case_sensitive;
        self
    }

    pub fn ignore_hidden(mut self, ignore_hidden: bool) -> Self {
        self.ignore_hidden = ignore_hidden;
        self
    }

    pub fn follow_symlinks(mut self, follow_symlinks: bool) -> Self {
        self.follow_symlinks = follow_symlinks;
        self
    }

    pub fn permission_errors(mut self, policy: PermissionErrorPolicy) -> Self {
        self.permission_errors = policy;
        self
    }
}

/// The filesystem kind of a path returned by a walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathKind {
    File,
    Directory,
    SymbolicLink,
    Other,
}

/// Where a returned path came from. This distinguishes direct entries from followed links.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathSource {
    DirectoryEntry,
    FollowedSymbolicLink,
}

/// A matched path together with the information needed to explain its origin.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FinalPath {
    pub path: PathBuf,
    pub relative_path: PathBuf,
    pub kind: PathKind,
    pub source: PathSource,
    pub symlink_target: Option<PathBuf>,
}

/// Why a traversal decision ignored a path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IgnoreReason {
    Hidden,
    NegativePattern(usize),
    NoPositiveMatch,
}

/// Why the traverser stopped descending a directory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PruneReason {
    Hidden,
    NoPositivePattern,
    NegativePattern(usize),
    PermissionDenied,
    SymlinkCycle,
}

/// The action chosen for a symbolic link.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SymlinkAction {
    IncludedWithoutFollowing,
    Followed,
    PrunedCycle,
    IncludedUnresolved,
    Error,
}

/// A deterministic event emitted only by the explain entry point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExplainEvent {
    PatternCompiled {
        pattern_index: usize,
        pattern: String,
        negated: bool,
        case_sensitive: bool,
        path_separator: char,
        root_anchored: bool,
    },
    StartDirectory {
        root: PathBuf,
        canonical_root: PathBuf,
        path_separator: char,
        follow_symlinks: bool,
        ignore_hidden: bool,
    },
    DirectoryPruned {
        directory: PathBuf,
        relative_path: PathBuf,
        reason: PruneReason,
    },
    IgnoreHit {
        path: PathBuf,
        relative_path: PathBuf,
        reason: IgnoreReason,
    },
    SymlinkDecision {
        path: PathBuf,
        relative_path: PathBuf,
        action: SymlinkAction,
        target: Option<PathBuf>,
    },
    PathDecision {
        path: PathBuf,
        relative_path: PathBuf,
        included: bool,
        kind: PathKind,
        source: PathSource,
    },
}

/// A successful explain run. Every invocation has a unique identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExplainReport {
    pub invocation_id: u64,
    pub patterns: Vec<String>,
    pub start_directory: PathBuf,
    pub events: Vec<ExplainEvent>,
    pub results: Vec<FinalPath>,
}

/// Error returned by traversal entry points while preserving the existing [`PatternError`] type.
#[derive(Debug)]
pub enum WalkError {
    Pattern(PatternError),
    Io(io::Error),
}

impl PartialEq for WalkError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Pattern(left), Self::Pattern(right)) => left == right,
            (Self::Io(left), Self::Io(right)) => {
                left.kind() == right.kind() && left.raw_os_error() == right.raw_os_error()
            }
            _ => false,
        }
    }
}

impl Eq for WalkError {}

impl fmt::Display for WalkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pattern(error) => write!(formatter, "{error}"),
            Self::Io(error) => write!(formatter, "{error}"),
        }
    }
}

impl StdError for WalkError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Pattern(error) => Some(error),
            Self::Io(error) => Some(error),
        }
    }
}

impl From<PatternError> for WalkError {
    fn from(error: PatternError) -> Self {
        Self::Pattern(error)
    }
}

impl From<io::Error> for WalkError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

#[derive(Clone, Debug)]
struct CompiledPattern {
    source: String,
    normalized: String,
    negated: bool,
    root_anchored: bool,
    supports_prefix: bool,
    segments: Vec<String>,
}

impl CompiledPattern {
    fn matches(&self, relative_path: &str) -> bool {
        glob_match(self.normalized.as_bytes(), relative_path.as_bytes())
    }
}

impl<T: Trace> Engine<T> {
    fn new(
        root: impl AsRef<Path>,
        patterns: &[String],
        options: WalkOptions,
        mut trace: T,
    ) -> Result<Self, WalkError> {
        let invocation_id = NEXT_INVOCATION_ID.fetch_add(1, AtomicOrdering::Relaxed);
        let compiled = compile_patterns(patterns.iter().cloned(), &options)?;
        let root = root.as_ref().to_path_buf();
        let canonical_root = fs::canonicalize(&root)?;

        for (pattern_index, pattern) in compiled.patterns.iter().enumerate() {
            trace.record(ExplainEvent::PatternCompiled {
                pattern_index,
                pattern: pattern.source.clone(),
                negated: pattern.negated,
                case_sensitive: options.case_sensitive,
                path_separator: path_separator(),
                root_anchored: pattern.root_anchored,
            });
        }
        trace.record(ExplainEvent::StartDirectory {
            root: root.clone(),
            canonical_root: canonical_root.clone(),
            path_separator: path_separator(),
            follow_symlinks: options.follow_symlinks,
            ignore_hidden: options.ignore_hidden,
        });

        Ok(Self {
            root,
            canonical_root,
            options,
            compiled,
            invocation_id,
            trace,
            results: Vec::new(),
        })
    }

    fn relative_segments(relative_path: &Path) -> Vec<String> {
        relative_path
            .components()
            .map(|component| component.as_os_str().to_string_lossy().into_owned())
            .collect()
    }

    fn matching_pattern(&self, relative_path: &Path, negated: bool) -> Option<usize> {
        let key = relative_key(relative_path, self.options.case_sensitive);
        self.compiled
            .patterns
            .iter()
            .position(|pattern| pattern.negated == negated && pattern.matches(&key))
    }

    fn positive_match(&self, relative_path: &Path) -> Option<usize> {
        self.matching_pattern(relative_path, false)
    }

    fn negative_match(&self, relative_path: &Path) -> Option<usize> {
        self.matching_pattern(relative_path, true)
    }

    fn add_result(
        &mut self,
        path: PathBuf,
        relative_path: PathBuf,
        kind: PathKind,
        source: PathSource,
        symlink_target: Option<PathBuf>,
    ) {
        self.results.push(FinalPath {
            path,
            relative_path,
            kind,
            source,
            symlink_target,
        });
    }

    fn classify(
        &mut self,
        path: PathBuf,
        relative_path: PathBuf,
        kind: PathKind,
        source: PathSource,
    ) -> bool {
        if self.options.ignore_hidden && is_hidden_relative(&relative_path) {
            self.trace.record(ExplainEvent::IgnoreHit {
                path: path.clone(),
                relative_path: relative_path.clone(),
                reason: IgnoreReason::Hidden,
            });
            self.trace.record(ExplainEvent::PathDecision {
                path,
                relative_path,
                included: false,
                kind,
                source,
            });
            return false;
        }

        let positive = self.positive_match(&relative_path);
        let negative = self.negative_match(&relative_path);
        if let Some(pattern_index) = negative {
            self.trace.record(ExplainEvent::IgnoreHit {
                path: path.clone(),
                relative_path: relative_path.clone(),
                reason: IgnoreReason::NegativePattern(pattern_index),
            });
        } else if positive.is_none() {
            self.trace.record(ExplainEvent::IgnoreHit {
                path: path.clone(),
                relative_path: relative_path.clone(),
                reason: IgnoreReason::NoPositiveMatch,
            });
        }

        let included = positive.is_some() && negative.is_none();
        self.trace.record(ExplainEvent::PathDecision {
            path: path.clone(),
            relative_path: relative_path.clone(),
            included,
            kind,
            source,
        });
        if included {
            self.add_result(path, relative_path, kind, source, None);
        }
        included
    }

    fn no_positive_can_reach(&self, relative_path: &Path) -> bool {
        let segments = Self::relative_segments(relative_path);
        self.compiled.positives.iter().all(|index| {
            !positive_can_reach(&self.compiled.patterns[*index], &segments)
        })
    }

    fn negative_covers_directory(&self, relative_path: &Path) -> Option<usize> {
        let segments = Self::relative_segments(relative_path);
        self.compiled.negatives.iter().copied().find(|index| {
            negative_covers_descendants(&self.compiled.patterns[*index], &segments)
        })
    }

    fn handle_read_dir_error(
        &mut self,
        directory: &Path,
        relative_path: &Path,
        error: io::Error,
    ) -> Result<(), WalkError> {
        if error.kind() == io::ErrorKind::PermissionDenied
            && self.options.permission_errors == PermissionErrorPolicy::Skip
        {
            self.trace.record(ExplainEvent::DirectoryPruned {
                directory: directory.to_path_buf(),
                relative_path: relative_path.to_path_buf(),
                reason: PruneReason::PermissionDenied,
            });
            return Ok(());
        }
        Err(error.into())
    }

    fn walk(
        &mut self,
        directory: &Path,
        canonical_directory: &Path,
        relative_directory: &Path,
        ancestors: &HashSet<PathBuf>,
        inherited_source: PathSource,
    ) -> Result<(), WalkError> {
        let entries = match fs::read_dir(directory) {
            Ok(read_dir) => sorted_entries(read_dir)?,
            Err(error) => {
                return self.handle_read_dir_error(directory, relative_directory, error);
            }
        };

        for entry in entries {
            let path = entry.path();
            let relative_path = relative_directory.join(entry.file_name());
            let file_type = entry.file_type()?;

            if file_type.is_symlink() {
                self.visit_symlink(
                    path,
                    relative_path,
                    ancestors,
                    canonical_directory,
                    entry.file_name(),
                )?;
            } else {
                let kind = if file_type.is_dir() {
                    PathKind::Directory
                } else if file_type.is_file() {
                    PathKind::File
                } else {
                    PathKind::Other
                };
                if kind == PathKind::Directory {
                    self.visit_directory(
                        path,
                        canonical_directory.join(entry.file_name()),
                        relative_path,
                        ancestors,
                        inherited_source,
                    )?;
                } else {
                    self.classify(path, relative_path, kind, inherited_source);
                }
            }
        }
        Ok(())
    }

    fn descend(
        &mut self,
        path: &Path,
        canonical_path: &Path,
        relative_path: &Path,
        mut ancestors: HashSet<PathBuf>,
        source: PathSource,
    ) -> Result<(), WalkError> {
        if !ancestors.insert(canonical_path.to_path_buf()) {
            self.trace.record(ExplainEvent::SymlinkDecision {
                path: path.to_path_buf(),
                relative_path: relative_path.to_path_buf(),
                action: SymlinkAction::PrunedCycle,
                target: canonical_path.canonicalize().ok(),
            });
            self.trace.record(ExplainEvent::DirectoryPruned {
                directory: path.to_path_buf(),
                relative_path: relative_path.to_path_buf(),
                reason: PruneReason::SymlinkCycle,
            });
            return Ok(());
        }
        self.walk(path, canonical_path, relative_path, &ancestors, source)
    }

    fn visit_directory(
        &mut self,
        path: PathBuf,
        canonical_path: PathBuf,
        relative_path: PathBuf,
        ancestors: &HashSet<PathBuf>,
        source: PathSource,
    ) -> Result<(), WalkError> {
        if self.options.ignore_hidden && is_hidden_relative(&relative_path) {
            self.trace.record(ExplainEvent::IgnoreHit {
                path: path.clone(),
                relative_path: relative_path.clone(),
                reason: IgnoreReason::Hidden,
            });
            self.trace.record(ExplainEvent::DirectoryPruned {
                directory: path,
                relative_path,
                reason: PruneReason::Hidden,
            });
            return Ok(());
        }

        self.classify(
            path.clone(),
            relative_path.clone(),
            PathKind::Directory,
            source,
        );

        if let Some(pattern_index) = self.negative_covers_directory(&relative_path) {
            self.trace.record(ExplainEvent::DirectoryPruned {
                directory: path,
                relative_path,
                reason: PruneReason::NegativePattern(pattern_index),
            });
            return Ok(());
        }

        if self.no_positive_can_reach(&relative_path) {
            self.trace.record(ExplainEvent::DirectoryPruned {
                directory: path,
                relative_path,
                reason: PruneReason::NoPositivePattern,
            });
            return Ok(());
        }

        self.descend(
            &path,
            &canonical_path,
            &relative_path,
            ancestors.clone(),
            source,
        )
    }

    fn visit_symlink(
        &mut self,
        path: PathBuf,
        relative_path: PathBuf,
        ancestors: &HashSet<PathBuf>,
        canonical_parent: &Path,
        file_name: OsString,
    ) -> Result<(), WalkError> {
        let target = fs::read_link(&path).ok();
        if !self.options.follow_symlinks {
            self.trace.record(ExplainEvent::SymlinkDecision {
                path: path.clone(),
                relative_path: relative_path.clone(),
                action: SymlinkAction::IncludedWithoutFollowing,
                target: target.clone(),
            });
            self.classify(path, relative_path, PathKind::SymbolicLink, PathSource::DirectoryEntry);
            return Ok(());
        }

        match fs::metadata(&path) {
            Ok(metadata) => {
                if metadata.is_dir() {
                    let canonical_path = match fs::canonicalize(&path) {
                        Ok(canonical_path) => canonical_path,
                        Err(error) => return Err(error.into()),
                    };
                    if ancestors.contains(&canonical_path) {
                        self.trace.record(ExplainEvent::SymlinkDecision {
                            path: path.clone(),
                            relative_path: relative_path.clone(),
                            action: SymlinkAction::PrunedCycle,
                            target: target.clone(),
                        });
                        self.trace.record(ExplainEvent::DirectoryPruned {
                            directory: path,
                            relative_path,
                            reason: PruneReason::SymlinkCycle,
                        });
                        return Ok(());
                    }

                    self.trace.record(ExplainEvent::SymlinkDecision {
                        path: path.clone(),
                        relative_path: relative_path.clone(),
                        action: SymlinkAction::Followed,
                        target: target.clone(),
                    });
                    self.classify(
                        path.clone(),
                        relative_path.clone(),
                        PathKind::SymbolicLink,
                        PathSource::DirectoryEntry,
                    );

                    if self.options.ignore_hidden && is_hidden_relative(&relative_path) {
                        self.trace.record(ExplainEvent::DirectoryPruned {
                            directory: path,
                            relative_path,
                            reason: PruneReason::Hidden,
                        });
                        return Ok(());
                    }
                    if let Some(pattern_index) = self.negative_covers_directory(&relative_path) {
                        self.trace.record(ExplainEvent::DirectoryPruned {
                            directory: path,
                            relative_path,
                            reason: PruneReason::NegativePattern(pattern_index),
                        });
                        return Ok(());
                    }
                    if self.no_positive_can_reach(&relative_path) {
                        self.trace.record(ExplainEvent::DirectoryPruned {
                            directory: path,
                            relative_path,
                            reason: PruneReason::NoPositivePattern,
                        });
                        return Ok(());
                    }

                    self.descend(
                        &path,
                        &canonical_path,
                        &relative_path,
                        ancestors.clone(),
                        PathSource::FollowedSymbolicLink,
                    )
                } else {
                    self.trace.record(ExplainEvent::SymlinkDecision {
                        path: path.clone(),
                        relative_path: relative_path.clone(),
                        action: SymlinkAction::Followed,
                        target: target.clone(),
                    });
                    let kind = if metadata.is_file() {
                        PathKind::File
                    } else {
                        PathKind::Other
                    };
                    self.classify(path, relative_path, kind, PathSource::FollowedSymbolicLink);
                    Ok(())
                }
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                self.trace.record(ExplainEvent::SymlinkDecision {
                    path: path.clone(),
                    relative_path: relative_path.clone(),
                    action: SymlinkAction::IncludedUnresolved,
                    target: target.clone(),
                });
                self.classify(path, relative_path, PathKind::SymbolicLink, PathSource::DirectoryEntry);
                Ok(())
            }
            Err(error)
                if error.kind() == io::ErrorKind::PermissionDenied
                    && self.options.permission_errors == PermissionErrorPolicy::Skip =>
            {
                self.trace.record(ExplainEvent::SymlinkDecision {
                    path: path.clone(),
                    relative_path: relative_path.clone(),
                    action: SymlinkAction::Error,
                    target: target.clone(),
                });
                self.trace.record(ExplainEvent::DirectoryPruned {
                    directory: path,
                    relative_path,
                    reason: PruneReason::PermissionDenied,
                });
                Ok(())
            }
            Err(error) => Err(error.into()),
        }
    }

    fn run(mut self) -> Result<(u64, Vec<FinalPath>, Vec<ExplainEvent>), WalkError> {
        let mut ancestors = HashSet::new();
        ancestors.insert(self.canonical_root.clone());
        self.walk(
            &self.root.clone(),
            &self.canonical_root.clone(),
            Path::new(""),
            &ancestors,
            PathSource::DirectoryEntry,
        )?;
        Ok((self.invocation_id, self.results, self.trace.into_events()))
    }
}

impl NoTrace {
    fn into_events(self) -> Vec<ExplainEvent> {
        Vec::new()
    }
}

impl ExplainTrace {
    fn into_events(self) -> Vec<ExplainEvent> {
        self.events
    }
}

fn pattern_strings<I, P>(patterns: I) -> Vec<String>
where
    I: IntoIterator<Item = P>,
    P: AsRef<str>,
{
    patterns
        .into_iter()
        .map(|pattern| pattern.as_ref().to_owned())
        .collect()
}

/// Walks `root`, matching every encountered entry through the same matcher used by
/// [`crate::glob_match`]. Symlinks are not followed by default.
pub fn glob_walk<I, P>(
    root: impl AsRef<Path>,
    patterns: I,
    options: &WalkOptions,
) -> Result<Vec<FinalPath>, WalkError>
where
    I: IntoIterator<Item = P>,
    P: AsRef<str>,
{
    let patterns = pattern_strings(patterns);
    let engine = Engine::new(root, &patterns, options.clone(), NoTrace)?;
    let (_, results, _) = engine.run()?;
    Ok(results)
}

/// Walks `root` and returns the matched paths plus a deterministic explanation.
pub fn glob_walk_explain<I, P>(
    root: impl AsRef<Path>,
    patterns: I,
    options: &WalkOptions,
) -> Result<ExplainReport, WalkError>
where
    I: IntoIterator<Item = P>,
    P: AsRef<str>,
{
    let patterns = pattern_strings(patterns);
    let start_directory = root.as_ref().to_path_buf();
    let engine = Engine::new(root, &patterns, options.clone(), ExplainTrace::default())?;
    let (invocation_id, results, events) = engine.run()?;
    Ok(ExplainReport {
        invocation_id,
        patterns,
        start_directory,
        events,
        results,
    })
}

#[derive(Clone, Debug)]
struct CompiledPatterns {
    patterns: Vec<CompiledPattern>,
    positives: Vec<usize>,
    negatives: Vec<usize>,
}

fn split_pattern_segments(pattern: &str) -> Vec<String> {
    pattern.split('/').map(str::to_owned).collect()
}

fn brace_contains_slash(pattern: &str) -> bool {
    let mut depth = 0usize;
    let mut escaped = false;
    for byte in pattern.bytes() {
        if escaped {
            escaped = false;
            continue;
        }
        match byte {
            b'\\' => escaped = true,
            b'{' => depth += 1,
            b'}' => depth = depth.saturating_sub(1),
            b'/' if depth > 0 => return true,
            _ => {}
        }
    }
    false
}

fn compile_patterns<I, P>(patterns: I, options: &WalkOptions) -> Result<CompiledPatterns, WalkError>
where
    I: IntoIterator<Item = P>,
    P: Into<String>,
{
    let mut compiled = Vec::new();
    for pattern in patterns {
        let source = pattern.into();
        validate(&source)?;

        let mut normalized = source;
        let mut negated = false;
        while normalized.as_bytes().first() == Some(&b'!') {
            negated = !negated;
            normalized.remove(0);
        }

        let root_anchored = normalized.starts_with('/');
        if root_anchored {
            normalized.remove(0);
        }
        if !options.case_sensitive {
            normalized = String::from_utf8(
                normalized
                    .into_bytes()
                    .into_iter()
                    .map(u8::to_ascii_lowercase)
                    .collect(),
            )
            .expect("ASCII lowercasing preserves UTF-8");
        }

        let supports_prefix = !normalized.contains('\\')
            && !normalized.contains('[')
            && !brace_contains_slash(&normalized);
        let segments = split_pattern_segments(&normalized);
        compiled.push(CompiledPattern {
            source,
            normalized,
            negated,
            root_anchored,
            supports_prefix,
            segments,
        });
    }

    let positives = compiled
        .iter()
        .enumerate()
        .filter_map(|(index, pattern)| (!pattern.negated).then_some(index))
        .collect();
    let negatives = compiled
        .iter()
        .enumerate()
        .filter_map(|(index, pattern)| pattern.negated.then_some(index))
        .collect();

    Ok(CompiledPatterns {
        patterns: compiled,
        positives,
        negatives,
    })
}

fn segment_matches(pattern: &str, segment: &str) -> bool {
    glob_match(pattern.as_bytes(), segment.as_bytes())
}

fn positive_can_reach(compiled: &CompiledPattern, directory: &[String]) -> bool {
    if !compiled.supports_prefix || !compiled.positives_relevant() {
        return true;
    }

    let mut visited = HashSet::new();
    let mut pending = vec![(0usize, 0usize)];
    while let Some((glob_index, path_index)) = pending.pop() {
        if !visited.insert((glob_index, path_index)) {
            continue;
        }
        if path_index == directory.len() {
            return true;
        }
        if glob_index == compiled.segments.len() {
            continue;
        }

        let segment = &compiled.segments[glob_index];
        if segment == "**" {
            pending.push((glob_index + 1, path_index));
            pending.push((glob_index, path_index + 1));
        } else if segment_matches(segment, &directory[path_index]) {
            pending.push((glob_index + 1, path_index + 1));
        }
    }
    false
}

fn suffix_covers_every_descendant(segments: &[String]) -> bool {
    if !segments.iter().any(|segment| segment == "**") {
        return false;
    }
    segments.iter().filter(|segment| segment.as_str() != "**").all(|segment| segment == "*")
        && segments
            .iter()
            .filter(|segment| segment.as_str() == "*")
            .count()
            <= 1
}

fn negative_covers_descendants(
    compiled: &CompiledPattern,
    directory: &[String],
) -> bool {
    if !compiled.supports_prefix {
        return false;
    }

    let mut visited = HashSet::new();
    let mut pending = vec![(0usize, 0usize)];
    while let Some((glob_index, path_index)) = pending.pop() {
        if !visited.insert((glob_index, path_index)) {
            continue;
        }
        if path_index == directory.len()
            && suffix_covers_every_descendant(&compiled.segments[glob_index..])
        {
            return true;
        }
        if glob_index == compiled.segments.len() || path_index == directory.len() {
            continue;
        }

        let segment = &compiled.segments[glob_index];
        if segment == "**" {
            pending.push((glob_index + 1, path_index));
            pending.push((glob_index, path_index + 1));
        } else if segment_matches(segment, &directory[path_index]) {
            pending.push((glob_index + 1, path_index + 1));
        }
    }
    false
}

trait Trace {
    fn record(&mut self, event: ExplainEvent);
}

#[derive(Default)]
struct NoTrace;

impl Trace for NoTrace {
    fn record(&mut self, _event: ExplainEvent) {}
}

#[derive(Default)]
struct ExplainTrace {
    events: Vec<ExplainEvent>,
}

impl Trace for ExplainTrace {
    fn record(&mut self, event: ExplainEvent) {
        self.events.push(event);
    }
}

struct Engine<T: Trace> {
    root: PathBuf,
    canonical_root: PathBuf,
    options: WalkOptions,
    compiled: CompiledPatterns,
    invocation_id: u64,
    trace: T,
    results: Vec<FinalPath>,
}

fn path_separator() -> char {
    if cfg!(windows) {
        '\\'
    } else {
        '/'
    }
}

fn relative_key(relative_path: &Path, case_sensitive: bool) -> String {
    let value = relative_path
        .components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/");
    if case_sensitive {
        value
    } else {
        String::from_utf8(value.into_bytes().into_iter().map(u8::to_ascii_lowercase).collect())
            .expect("ASCII lowercasing preserves UTF-8")
    }
}

fn is_hidden_relative(relative_path: &Path) -> bool {
    relative_path
        .components()
        .any(|component| component.as_os_str().to_string_lossy().starts_with('.'))
}

fn sorted_entries(read_dir: ReadDir) -> Result<Vec<DirEntry>, io::Error> {
    let mut entries = read_dir.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by(|left, right| left.file_name().cmp(&right.file_name()));
    Ok(entries)
}

impl CompiledPattern {
    fn positives_relevant(&self) -> bool {
        !self.negated
    }
}
