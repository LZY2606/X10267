//! Deterministic filesystem traversal and explain support.

//! This module is intentionally stateless: each call compiles its patterns,
//! performs one deterministic traversal, and takes no data from prior calls.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{glob_match, validate, Error as GlobError};

static NEXT_INVOCATION: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InvocationId(u64);

impl fmt::Display for InvocationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:016x}", self.0)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum WalkErrorKind {
    InvalidPattern(GlobError),
    Io,
}

#[derive(Debug)]
pub struct WalkError {
    pub invocation_id: InvocationId,
    pub kind: WalkErrorKind,
    pub path: Option<PathBuf>,
    source: Option<io::Error>,
}

impl WalkError {
    pub fn pattern_error(&self) -> Option<GlobError> {
        match self.kind {
            WalkErrorKind::InvalidPattern(error) => Some(error),
            WalkErrorKind::Io => None,
        }
    }

    pub fn io_error(&self) -> Option<&io::Error> {
        self.source.as_ref()
    }
}

impl fmt::Display for WalkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt:: Result {
        match self.kind {
            WalkErrorKind::InvalidPattern(error) => write!(f, "invalid glob pattern: {error}"),
            WalkErrorKind::Io => match &self.path {
                Some(path) => write!(f, "I/O error while walking {}", path.display()),
                None => f.write_str("I/O error while walking"),
            },
        }
    }
}

impl Error for WalkError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source.as_ref().map(|error| error as &(dyn Error + 'static))
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PermissionPolicy {
    #[default]
    Stop,
    Continue,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PathSeparator {
    ForwardSlash,
    Native,
}

impl Default for PathSeparator {
    fn default() -> Self {
        Self::Native
    }
}

#[derive(Clone, Debug)]
pub struct WalkOptions {
    pub include_hidden: bool,
    pub follow_symlinks: bool,
    pub case_sensitive: bool,
    pub permission_policy: PermissionPolicy,
    pub path_separator: PathSeparator,
}

impl Default for WalkOptions {
    fn default() -> Self {
        Self {
            include_hidden: false,
            follow_symlinks: false,
            case_sensitive: true,
            permission_policy: PermissionPolicy::Stop,
            path_separator: PathSeparator::Native,
        }
    }
}

impl WalkOptions {
    pub fn include_hidden(mut self, include_hidden: bool) -> Self {
        self.include_hidden = include_hidden;
        self
    }

    pub fn follow_symlinks(mut self, follow_symlinks: bool) -> Self {
        self.follow_symlinks = follow_symlinks;
        self
    }

    pub fn case_sensitive(mut self, case_sensitive: bool) -> Self {
        self.case_sensitive = case_sensitive;
        self
    }

    pub fn permission_policy(mut self, policy: PermissionPolicy) -> Self {
        self.permission_policy = policy;
        self
    }

    pub fn path_separator(mut self, separator: PathSeparator) -> Self {
        self.path_separator = separator;
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledPattern {
    pub index: usize,
    pub pattern: String,
    pub negated: bool,
    pub literal_dirs: Vec<String>,
    pub separator: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PatternCompileTrace {
    pub index: usize,
    pub original: String,
    pub negated: bool,
    pub effective: String,
    pub literal_dirs: Vec<String>,
    pub separator: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StartDirectoryTrace {
    pub root: PathBuf,
    pub relative: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PruneReason {
    Hidden,
    NoPatternCanMatch,
    NoPositivePatternCanMatch,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectoryPruneTrace {
    pub path: PathBuf,
    pub relative: String,
    pub reason: PruneReason,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IgnoreTrace {
    pub path: PathBuf,
    pub relative: String,
    pub pattern_index: usize,
    pub pattern: String,
    pub directory_scope: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SymlinkDecision {
    NotFollowed,
    FollowedDirectory,
    FollowedFile,
    Loop,
    Broken,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SymlinkTrace {
    pub path: PathBuf,
    pub relative: String,
    pub decision: SymlinkDecision,
    pub target: Option<PathBuf>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PermissionTrace {
    pub path: PathBuf,
    pub relative: String,
    pub policy: PermissionPolicy,
    pub message: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CaseCollisionTrace {
    pub normalized: String,
    pub pattern_indices: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PathSource {
    File,
    Symlink,
    FollowedSymlink,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FinalPathTrace {
    pub path: PathBuf,
    pub relative: String,
    pub source: PathSource,
    pub matched_pattern_index: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExcludedPathTrace {
    pub path: PathBuf,
    pub relative: String,
    pub matched_negative_pattern: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TraceEvent {
    PatternCompiled(PatternCompileTrace),
    StartDirectory(StartDirectoryTrace),
    DirectoryPruned(DirectoryPruneTrace),
    IgnoreHit(IgnoreTrace),
    SymlinkDecision(SymlinkTrace),
    PermissionDenied(PermissionTrace),
    CaseCollision(CaseCollisionTrace),
    FinalPath(FinalPathTrace),
    ExcludedPath(ExcludedPathTrace),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExplainPath {
    pub path: PathBuf,
    pub relative: String,
    pub source: PathSource,
    pub matched_pattern_index: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExplainReport {
    pub invocation_id: InvocationId,
    pub paths: Vec<ExplainPath>,
    pub trace: Vec<TraceEvent>,
}

#[derive(Clone)]
struct CompiledInput {
    index: usize,
    original: String,
    effective: Vec<u8>,
    negated: bool,
    literal_dirs: Vec<String>,
    absolute: bool,
}

#[derive(Clone)]
struct Engine<'a> {
    options: WalkOptions,
    patterns: Vec<CompiledInput>,
    trace: Vec<TraceEvent>,
    paths: Vec<ExplainPath>,
    invocation_id: InvocationId,
}

pub fn walk<I, S>(root: impl AsRef<Path>, patterns: I, options: WalkOptions) -> Result<Vec<PathBuf>, WalkError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    explain(root, patterns, options).map(|report| report.paths.into_iter().map(|path| path.path).collect())
}

pub fn explain_walk<I, S>(
    root: impl AsRef<Path>,
    patterns: I,
    options: WalkOptions,
) -> Result<ExplainReport, WalkError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    explain(root, patterns, options)
}

fn explain<I, S>(root: impl AsRef<Path>, patterns: I, options: WalkOptions) -> Result<ExplainReport, WalkError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let invocation_id = InvocationId(NEXT_INVOCATION.fetch_add(1, Ordering::Relaxed));
    let root = root.as_ref();
    let compiled = compile_patterns(patterns, &options, invocation_id)?;
    let mut engine = Engine {
        options,
        patterns: compiled,
        trace: Vec::new(),
        paths: Vec::new(),
        invocation_id,
    };

    for pattern in &engine.patterns {
        engine.trace.push(TraceEvent::PatternCompiled(PatternCompileTrace {
            index: pattern.index,
            original: pattern.original.clone(),
            negated: pattern.negated,
            effective: String::from_utf8_lossy(&pattern.effective).into_owned(),
            literal_dirs: pattern.literal_dirs.clone(),
            separator: engine.separator_label().to_owned(),
        }));
    }
    engine.record_case_collisions();
    engine.trace.push(TraceEvent::StartDirectory(StartDirectoryTrace {
        root: root.to_path_buf(),
        relative: String::new(),
    }));
    if !engine.has_positive_pattern() {
        engine.trace.push(TraceEvent::DirectoryPruned(DirectoryPruneTrace {
            path: root.to_path_buf(),
            relative: String::new(),
            reason: PruneReason::NoPositivePatternCanMatch,
        }));
        return Ok(ExplainReport {
            invocation_id,
            paths: Vec::new(),
            trace: engine.trace,
        });
    }

    engine.visit_directory(root, String::new(), Vec::new())?;
    Ok(ExplainReport {
        invocation_id,
        paths: engine.paths,
        trace: engine.trace,
    })
}

fn compile_patterns<I, S>(
    patterns: I,
    options: &WalkOptions,
    invocation_id: InvocationId,
) -> Result<Vec<CompiledInput>, WalkError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    patterns
        .into_iter()
        .enumerate()
        .map(|(index, pattern)| {
            let original = pattern.as_ref().to_owned();
            validate(original.as_bytes()).map_err(|error| WalkError {
                invocation_id,
                kind: WalkErrorKind::InvalidPattern(error),
                path: None,
                source: None,
            })?;

            let bytes = original.as_bytes();
            let bangs = bytes.iter().take_while(|byte| **byte == b'!').count();
            let mut effective = bytes[bangs..].to_vec();
            normalize_separators(&mut effective, options.path_separator);
            let mut compared = effective.clone();
            if !options.case_sensitive {
                compared.make_ascii_lowercase();
            }
            let literal_dirs = literal_dirs(&effective);
            let absolute = matches!(effective.first(), Some(b'/'));

            Ok(CompiledInput {
                index,
                original,
                effective: compared,
                negated: bangs % 2 == 1,
                literal_dirs,
                absolute,
            })
        })
        .collect()
}

impl<'a> Engine<'a> {
    fn separator_label(&self) -> &'static str {
        if cfg!(windows) && self.options.path_separator == PathSeparator::Native {
            "/ or \\"
        } else {
            "/"
        }
    }

    fn record_case_collisions(&mut self) {
        if self.options.case_sensitive {
            return;
        }

        let mut groups = BTreeMap::<Vec<u8>, BTreeMap<Vec<u8>, Vec<usize>>>::new();
        for pattern in &self.patterns {
            groups
                .entry(pattern.effective.to_ascii_lowercase())
                .or_default()
                .entry(pattern.effective.clone())
                .or_default()
                .push(pattern.index);
        }
        for (normalized, variants) in groups {
            if variants.len() > 1 {
                let indices = variants.into_values().flatten().collect();
                self.trace.push(TraceEvent::CaseCollision(CaseCollisionTrace {
                    normalized: String::from_utf8_lossy(&normalized).into_owned(),
                    pattern_indices: indices,
                }));
            }
        }
    }

    fn has_positive_pattern(&self) -> bool {
        self.patterns
            .iter()
            .any(|pattern| !pattern.negated && !pattern.absolute)
    }

    fn visit_directory(
        &mut self,
        directory: &Path,
        relative: String,
        mut real_ancestors: Vec<PathBuf>,
    ) -> Result<(), WalkError> {
        let read_result = fs::read_dir(directory);
        let mut entries = match read_result {
            Ok(entries) => entries,
            Err(error) => return self.handle_io_error(directory, relative, error),
        };

        let mut sorted = Vec::new();
        loop {
            match entries.next() {
                Some(Ok(entry)) => sorted.push(entry),
                Some(Err(error)) => {
                    let entry_path = directory.join("<unreadable-entry>");
                    self.handle_io_error(&entry_path, relative.clone(), error)?;
                }
                None => break,
            }
        }
        sorted.sort_by_key(|entry| entry.file_name());

        for entry in sorted {
            self.visit_entry(directory, &relative, entry, &mut real_ancestors)?;
        }
        Ok(())
    }

    fn visit_entry(
        &mut self,
        parent: &Path,
        parent_relative: &str,
        entry: fs::DirEntry,
        real_ancestors: &mut Vec<PathBuf>,
    ) -> Result<(), WalkError> {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            return Ok(());
        };
        if !self.options.include_hidden && name.starts_with('.') {
            let relative = join_relative(parent_relative, name);
            let is_directory = entry.file_type().map(|file_type| file_type.is_dir()).unwrap_or(false);
            if is_directory {
                self.trace.push(TraceEvent::DirectoryPruned(DirectoryPruneTrace {
                    path: entry.path(),
                    relative,
                    reason: PruneReason::Hidden,
                }));
            } else {
                self.trace.push(TraceEvent::ExcludedPath(ExcludedPathTrace {
                    path: entry.path(),
                    relative,
                    matched_negative_pattern: None,
                }));
            }
            return Ok(());
        }

        let path = parent.join(name);
        let relative = join_relative(parent_relative, name);
        let file_type = match entry.file_type() {
            Ok(file_type) => file_type,
            Err(error) => return self.handle_io_error(&path, relative, error),
        };

        if file_type.is_symlink() {
            let target = fs::read_link(&path).ok();
            if self.options.follow_symlinks {
                self.follow_symlink(&path, relative, target, real_ancestors)
            } else {
                self.trace.push(TraceEvent::SymlinkDecision(SymlinkTrace {
                    path: path.clone(),
                    relative: relative.clone(),
                    decision: SymlinkDecision::NotFollowed,
                    target,
                }));
                self.classify_final_entry(&path, relative, PathSource::Symlink);
                Ok(())
            }
        } else if file_type.is_dir() {
            self.visit_real_directory(&path, relative, real_ancestors)
        } else {
            self.classify_final_entry(&path, relative, PathSource::File);
            Ok(())
        }
    }

    fn visit_real_directory(
        &mut self,
        path: &Path,
        relative: String,
        real_ancestors: &mut Vec<PathBuf>,
    ) -> Result<(), WalkError> {
        if let Some((pattern_index, pattern)) = self.negative_scope(&relative) {
            self.trace.push(TraceEvent::IgnoreHit(IgnoreTrace {
                path: path.to_path_buf(),
                relative,
                pattern_index,
                pattern,
                directory_scope: true,
            }));
            return Ok(());
        }

        if !self.positive_can_descend(&relative) {
            self.trace.push(TraceEvent::DirectoryPruned(DirectoryPruneTrace {
                path: path.to_path_buf(),
                relative,
                reason: PruneReason::NoPositivePatternCanMatch,
            }));
            return Ok(());
        }

        self.visit_directory(path, relative, real_ancestors.clone())
    }

    fn follow_symlink(
        &mut self,
        path: &Path,
        relative: String,
        target: Option<PathBuf>,
        real_ancestors: &mut Vec<PathBuf>,
    ) -> Result<(), WalkError> {
        let metadata = match fs::metadata(path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                self.trace.push(TraceEvent::SymlinkDecision(SymlinkTrace {
                    path: path.to_path_buf(),
                    relative: relative.clone(),
                    decision: SymlinkDecision::Broken,
                    target,
                }));
                self.classify_final_entry(path, relative, PathSource::Symlink);
                return Ok(());
            }
            Err(error) => return self.handle_io_error(path, relative, error),
        };

        let canonical = match fs::canonicalize(path) {
            Ok(canonical) => canonical,
            Err(error) => return self.handle_io_error(path, relative, error),
        };

        if real_ancestors.iter().any(|ancestor| ancestor == &canonical) {
            self.trace.push(TraceEvent::SymlinkDecision(SymlinkTrace {
                path: path.to_path_buf(),
                relative,
                decision: SymlinkDecision::Loop,
                target,
            }));
            return Ok(());
        }

        if metadata.is_dir() {
            self.trace.push(TraceEvent::SymlinkDecision(SymlinkTrace {
                path: path.to_path_buf(),
                relative: relative.clone(),
                decision: SymlinkDecision::FollowedDirectory,
                target,
            }));
            let mut next_ancestors = real_ancestors.clone();
            next_ancestors.push(canonical);
            self.visit_directory(path, relative, next_ancestors)
        } else {
            self.trace.push(TraceEvent::SymlinkDecision(SymlinkTrace {
                path: path.to_path_buf(),
                relative: relative.clone(),
                decision: SymlinkDecision::FollowedFile,
                target,
            }));
            self.classify_final_entry(path, relative, PathSource::FollowedSymlink);
            Ok(())
        }
    }

    fn classify_final_entry(&mut self, path: &Path, relative: String, source: PathSource) {
        let compared = self.comparison_path(&relative);
        if let Some(index) = self.negative_match(&compared) {
            self.trace.push(TraceEvent::IgnoreHit(IgnoreTrace {
                path: path.to_path_buf(),
                relative,
                pattern_index: index,
                pattern: self.patterns[index].original.clone(),
                directory_scope: false,
            }));
            return;
        }

        if let Some(index) = self.positive_match(&compared) {
            self.paths.push(ExplainPath {
                path: path.to_path_buf(),
                relative: relative.clone(),
                source,
                matched_pattern_index: index,
            });
            self.trace.push(TraceEvent::FinalPath(FinalPathTrace {
                path: path.to_path_buf(),
                relative,
                source,
                matched_pattern_index: index,
            }));
        } else {
            self.trace.push(TraceEvent::ExcludedPath(ExcludedPathTrace {
                path: path.to_path_buf(),
                relative,
                matched_negative_pattern: None,
            }));
        }
    }

    fn comparison_path(&self, relative: &str) -> Vec<u8> {
        let mut bytes = relative.as_bytes().to_vec();
        normalize_separators(&mut bytes, self.options.path_separator);
        if !self.options.case_sensitive {
            bytes.make_ascii_lowercase();
        }
        bytes
    }

    fn positive_match(&self, path: &[u8]) -> Option<usize> {
        self.patterns
            .iter()
            .find(|pattern| !pattern.negated && !pattern.absolute && glob_match(&pattern.effective, path))
            .map(|pattern| pattern.index)
    }

    fn negative_match(&self, path: &[u8]) -> Option<usize> {
        self.patterns
            .iter()
            .find(|pattern| pattern.negated && !pattern.absolute && glob_match(&pattern.effective, path))
            .map(|pattern| pattern.index)
    }

    fn negative_scope(&self, relative: &str) -> Option<(usize, String)> {
        let compared = self.comparison_path(relative);
        self.patterns
            .iter()
            .filter(|pattern| pattern.negated && !pattern.absolute)
            .find(|pattern| negative_covers_dir(&pattern.effective, &compared))
            .map(|pattern| (pattern.index, pattern.original.clone()))
    }

    fn positive_can_descend(&self, relative: &str) -> bool {
        let compared = self.comparison_path(relative);
        self.patterns
            .iter()
            .any(|pattern| !pattern.negated && !pattern.absolute && pattern_can_descend(pattern, &compared))
    }

    fn handle_io_error(
        &mut self,
        path: &Path,
        relative: String,
        error: io::Error,
    ) -> Result<(), WalkError> {
        let denied = error.kind() == io::ErrorKind::PermissionDenied;
        if denied && self.options.permission_policy == PermissionPolicy::Continue {
            self.trace.push(TraceEvent::PermissionDenied(PermissionTrace {
                path: path.to_path_buf(),
                relative,
                policy: self.options.permission_policy,
                message: error.to_string(),
            }));
            Ok(())
        } else {
            Err(WalkError {
                invocation_id: self.invocation_id,
                kind: WalkErrorKind::Io,
                path: Some(path.to_path_buf()),
                source: Some(error),
            })
        }
    }
}

fn join_relative(parent: &str, name: &str) -> String {
    if parent.is_empty() {
        name.to_owned()
    } else {
        format!("{parent}/{name}")
    }
}

fn normalize_separators(bytes: &mut Vec<u8>, separator: PathSeparator) {
    if cfg!(windows) && separator == PathSeparator::Native {
        for byte in bytes.iter_mut() {
            if *byte == b'\\' {
                *byte = b'/';
            }
        }
    }
}

fn literal_dirs(pattern: &[u8]) -> Vec<String> {
    let mut dirs = Vec::new();
    let mut current = Vec::new();
    let mut index = 0;
    if pattern.first() == Some(&b'/') {
        index = 1;
    }

    while index < pattern.len() {
        match pattern[index] {
            b'\\' => return dirs,
            b'/' => {
                if !current.is_empty() {
                    dirs.push(String::from_utf8_lossy(&current).into_owned());
                    current.clear();
                }
            }
            b'*' | b'?' | b'[' | b'{' => return dirs,
            _ => current.push(pattern[index]),
        }
        index += 1;
    }
    dirs
}

fn pattern_can_descend(pattern: &CompiledInput, directory: &[u8]) -> bool {
    for (index, prefix) in pattern.literal_dirs.iter().enumerate() {
        let actual = match literal_segment(directory, index) {
            Some(segment) => segment,
            None => return false,
        };
        if actual != prefix.as_bytes() {
            return false;
        }
    }
    true
}

fn literal_segment(path: &[u8], wanted: usize) -> Option<&[u8]> {
    let mut start = 0;
    for index in 0..wanted {
        let end = path.get(start..)?.iter().position(|byte| *byte == b'/')?;
        start += end + 1;
        let _ = index;
    }
    let rest = path.get(start..)?;
    match rest.iter().position(|byte| *byte == b'/') {
        Some(end) => rest.get(..end),
        None => Some(rest),
    }
}

fn negative_covers_dir(pattern: &[u8], directory: &[u8]) -> bool {
    if glob_match(pattern, directory) {
        return true;
    }
    for suffix in [b"/**".as_slice(), b"/*".as_slice(), b"/**/".as_slice()] {
        if let Some(base) = pattern.strip_suffix(suffix) {
            if directory == base || directory.starts_with(&[base, b"/".as_slice()].concat()) {
                return true;
            }
        }
    }
    false
}
