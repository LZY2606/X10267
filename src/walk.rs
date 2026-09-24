//! Deterministic directory traversal built on top of the [`glob_match`] engine.
//!
//! [`Walker`] walks one or more root directories and matches every entry
//! against a set of positive and negative glob patterns using the same
//! [`glob_match`]/[`validate`] core that the rest of the crate exposes — it
//! never re-implements pattern matching. Every [`Walker::run`] call is
//! independent: no cache, visited set, counter or other state ever leaks
//! between calls, and every call carries a fresh, monotonically increasing
//! [`RunId`].
//!
//! With [`Walker::explain`] the same traversal additionally returns an
//! [`ExplainTrace`]: a deterministic record of pattern compilation, start
//! directories, directory pruning, ignore hits, symlink decisions and the
//! reason every included or excluded path got its verdict. The trace is
//! sorted by a fixed (root, path, phase) key, so it is byte-for-byte
//! identical under different thread scheduling.
use std::cmp::Ordering;
use std::collections::HashSet;
use std::error::Error as StdError;
use std::fmt;
use std::fs::{self, DirEntry, Metadata, ReadDir};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

use crate::{glob_match, validate, Error as GlobError};

/// Policy for errors encountered while reading a directory.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorPolicy {
    /// Collect the failure in [`WalkOutput::errors`] and continue with the
    /// remaining directories. This is the default.
    #[default]
    Continue,
    /// Stop the walk as soon as any directory cannot be read and return
    /// [`WalkError::Aborted`].
    Abort,
}

/// Policy for entries whose names start with a dot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum HiddenPolicy {
    /// Hidden files and directories never match and hidden directories are
    /// not descended into. This is the default.
    #[default]
    Skip,
    /// Hidden entries are treated like any other entry.
    Include,
}

/// Policy for symbolic links encountered during the walk.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum SymlinkPolicy {
    /// Never follow a symbolic link. A link is evaluated as an entry and
    /// never descended into, so no link can form a traversal cycle. This is
    /// the default.
    #[default]
    Skip,
    /// Follow directory links. Links that resolve to a directory already on
    /// the current walk (canonical path) are reported with
    /// [`SymlinkSkip::Loop`] and are not followed again.
    Follow,
}

/// Whether matching is case sensitive.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum CaseSensitivity {
    /// Case sensitive matching (e.g. `File.rs` does not match `file.rs`).
    /// This is the default on every platform.
    #[default]
    Sensitive,
    /// ASCII case-insensitive matching: patterns and entry names are folded
    /// to lower case (ASCII only) before reaching the matcher.
    Insensitive,
}

/// What a directory entry was classified as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EntryKind {
    /// A regular file.
    File,
    /// A directory, whether it was descended into or pruned.
    Directory,
    /// A symbolic link.
    Symlink,
    /// Any other file type (fifo, socket, device, ...).
    Other,
}

/// Why a directory subtree was not descended into.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PruneReason {
    /// The entry name starts with a dot and [`HiddenPolicy::Skip`] is set.
    Hidden,
    /// No positive pattern can match anything beneath the directory.
    /// `prefix` is the literal glob prefix that constrained the subtree.
    NoPositiveMatch {
        /// The deepest literal prefix shared by the matching patterns.
        prefix: String,
    },
    /// The last pattern matching the directory is a negative one whose
    /// trailing `/**` excludes the whole subtree (e.g. `!keep/**`).
    IgnoredSubtree {
        /// Index of the negative pattern as passed to [`Walker::new`].
        pattern_index: usize,
    },
    /// A symbolic link whose target is already part of the walk.
    SymlinkLoop,
}

/// Why an examined entry did not end up in the result set.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExcludeReason {
    /// No positive pattern matched the entry.
    NoPositiveMatch,
    /// The last matching pattern was negative.
    Ignored {
        /// Index of the negative pattern.
        pattern_index: usize,
    },
    /// The entry is a directory and only files were requested.
    DirectoryWhileFilesOnly,
}

/// Why a symbolic link was not followed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SymlinkSkip {
    /// [`SymlinkPolicy::Skip`] is configured.
    Policy,
    /// The link resolves to an ancestor directory of the walk, forming a
    /// cycle.
    Loop,
    /// The link target does not exist or cannot be read.
    Dangling,
}

/// A failure reading one directory, presented according to [`ErrorPolicy`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct IoFailure {
    /// The directory (relative to its root) that could not be read.
    pub relative_dir: String,
    /// The root the directory belonged to.
    pub root: String,
    /// Stable classification of the OS error.
    pub kind: IoErrorKind,
    /// The OS error message.
    pub message: String,
}

/// Portable classification of a directory read failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum IoErrorKind {
    /// A missing directory.
    NotFound,
    /// A permission denial.
    PermissionDenied,
    /// Any other I/O error.
    Other,
    /// A failure deliberately injected through [`Fault`].
    Injected,
}

/// An error that prevents a walk from producing a complete result.
#[derive(Debug)]
#[non_exhaustive]
pub enum WalkError {
    /// A glob pattern did not pass [`validate`].
    InvalidPattern {
        /// Index of the offending pattern as passed to [`Walker::new`].
        index: usize,
        /// The validation error from the existing error layer.
        source: GlobError,
    },
    /// The start directory itself could not be canonicalized or read.
    InvalidRoot {
        /// The root as passed by the caller.
        root: String,
        /// Portable classification.
        kind: IoErrorKind,
        /// The OS error message.
        message: String,
    },
    /// [`ErrorPolicy::Abort`] stopped the walk after this failure.
    Aborted(IoFailure),
}

/// A one-shot fault that makes exactly one directory read fail.
///
/// This is part of the public API specifically so callers can test their
/// handling of [`ErrorPolicy`] and verify recovery: once the fault has fired
/// it is consumed, and a repeated call with the same [`Walker`] succeeds.
/// It never fires on a walker without the fault installed.
#[derive(Debug)]
pub struct Fault {
    /// Relative directory (forward-slash form, `""` for the root) whose
    /// next read fails.
    pub relative_dir: String,
    remaining: Arc<AtomicUsize>,
}

/// Identity of a single [`Walker::run`] / [`Walker::explain`] invocation.
///
/// Every traversal allocates a fresh id from a process-wide monotonic
/// counter. It is the "version identity" of the run: all internal counters,
/// visited sets and buffers live and die with that id, so comparing traces
/// from two runs proves nothing is reused from a previous (possibly failed)
/// call.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RunId(u64);

static NEXT_RUN_ID: AtomicU64 = AtomicU64::new(1);

impl RunId {
    fn fresh() -> RunId {
        RunId(NEXT_RUN_ID.fetch_add(1, AtomicOrdering::Relaxed))
    }

    /// The monotonic numeric identity of the run.
    #[must_use]
    pub fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for IoFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "could not read '{}' under '{}': {}",
            self.relative_dir, self.root, self.message
        )
    }
}

impl StdError for IoFailure {}

impl fmt::Display for WalkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt:: Result {
        match self {
            WalkError::InvalidPattern { index, source } => {
                write!(f, "pattern at index {index} is invalid: {source}")
            }
            WalkError::InvalidRoot { root, message, .. } => {
                write!(f, "start directory {root:?} is not readable: {message}")
            }
            WalkError::Aborted(failure) => write!(f, "walk aborted: {failure}"),
        }
    }
}

impl StdError for WalkError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            WalkError::InvalidPattern { source, .. } => Some(source),
            WalkError::Aborted(failure) => Some(failure),
            WalkError::InvalidRoot { .. } => None,
        }
    }
}

/// One recorded step of an explain run.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Event {
    /// The root the event belongs to (as passed to [`Walker::explain`]).
    pub root: String,
    /// Entry path relative to `root`, always `/`-separated. Empty for
    /// events about the root itself.
    pub relative: String,
    /// What happened.
    pub kind: EventKind,
}

/// The recorded decision of one explain event.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EventKind {
    /// A pattern was compiled: validated, negation stripped, literal prefix
    /// extracted. Emitted once per pattern before any directory is read.
    PatternCompiled {
        /// Index of the pattern as passed to [`Walker::new`].
        index: usize,
        /// The pattern text after normalization.
        pattern: String,
        /// Whether the pattern is negative (starts with `!`).
        negated: bool,
        /// Literal leading path segments used for subtree pruning.
        literal_prefix: Vec<String>,
    },
    /// Traversal entered a start directory.
    StartDir,
    /// A directory subtree was not descended into.
    DirPruned {
        /// Why the subtree was skipped.
        reason: PruneReason,
    },
    /// A negative pattern matched an entry.
    IgnoreHit {
        /// Index of the negative pattern.
        pattern_index: usize,
    },
    /// A symbolic link was classified and its follow decision made.
    SymlinkDecision {
        /// Whether the link was followed into its target directory.
        followed: bool,
        /// Why it was not followed, if it was not.
        skipped: Option<SymlinkSkip>,
    },
    /// An entry got its final verdict.
    EntryEvaluated {
        /// What the entry is.
        kind: EntryKind,
        /// Whether the entry is part of the result set.
        included: bool,
        /// Indices of the positive patterns that matched, in pattern order.
        matched_by: Vec<usize>,
        /// Why the entry was excluded, if it was.
        excluded: Option<ExcludeReason>,
    },
    /// A directory could not be read; recorded under both error policies.
    ReadFailed {
        /// The failure, as it also appears in [`WalkOutput::errors`].
        failure: IoFailure,
    },
}

/// Deterministic record of one explained run.
///
/// Events are sorted by `(root, relative path, phase)` where the phase
/// orders events about the same entry (compile < start < prune < ignore <
/// symlink < verdict < read failure). Two runs over an unchanged tree with
/// the same walker produce identical traces regardless of thread count or
/// scheduling.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ExplainTrace {
    /// Identity of the run that produced this trace.
    pub run_id: RunId,
    /// All recorded events in deterministic order.
    pub events: Vec<Event>,
}

impl ExplainTrace {
    /// All events concerning `relative` under `root`, in order.
    #[must_use]
    pub fn events_for(&self, root: &str, relative: &str) -> Vec<&Event> {
        self.events
            .iter()
            .filter(|event| event.root == root && event.relative == relative)
            .collect()
    }

    /// The recorded verdict for `relative` under `root`, if it was examined.
    #[must_use]
    pub fn verdict(&self, root: &str, relative: &str) -> Option<&EventKind> {
        self.events_for(root, relative).iter().find_map(|event| match &event.kind {
            kind @ EventKind::EntryEvaluated { .. } => Some(kind),
            _ => None,
        })
    }
}

/// The result of one [`Walker::run`] / [`Walker::explain`] call.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct WalkOutput {
    /// Identity of this run.
    pub run_id: RunId,
    /// Matched paths, sorted lexicographically by `(root, relative path)`.
    pub paths: Vec<PathBuf>,
    /// Directory read failures collected under [`ErrorPolicy::Continue`].
    pub errors: Vec<IoFailure>,
    /// The explain trace, present only for [`Walker::explain`].
    pub explain: Option<ExplainTrace>,
}

/// A pattern after compilation: negation stripped, separators normalized,
/// literal prefix extracted for pruning.
#[derive(Clone, Debug)]
struct CompiledPattern {
    index: usize,
    /// Pattern text with any leading `!` removed and `\` separators
    /// normalized to `/` (escaped characters are preserved by the
    /// normalization below).
    body: String,
    negated: bool,
    /// Leading segments that contain no glob magic, used to prove that a
    /// directory subtree cannot contain any match.
    literal_prefix: Vec<String>,
    /// Body without a trailing `/**`, when present. A negative pattern of
    /// this shape excludes the whole subtree rooted at its prefix.
    subtree_root: Option<String>,
}

impl CompiledPattern {
    fn compile(index: usize, raw: &str) -> Result<CompiledPattern, WalkError> {
        validate(raw).map_err(|source| WalkError::InvalidPattern { index, source })?;

        let mut body = raw;
        let mut negated = false;
        while let Some(rest) = body.strip_prefix('!') {
            negated = !negated;
            body = rest;
        }

        // `\` is a path separator on Windows and an escape in glob syntax.
        // The matcher treats both `/` and `\` as separators in the path, so
        // an unescaped `\` in the pattern is normalized to `/` here; escaped
        // characters (`\\`, `\{`, ...) keep their backslash.
        let mut normalized = String::with_capacity(body.len());
        let mut chars = body.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.peek() {
                    Some(&next) if is_glob_special(next) => {
                        normalized.push(c);
                        normalized.push(next);
                        chars.next();
                    }
                    _ => normalized.push('/'),
                }
            } else {
                normalized.push(c);
            }
        }

        let subtree_root = normalized
            .strip_suffix("/**")
            .filter(|prefix| !prefix.is_empty() && !prefix.contains(['*', '?', '[', '{']))
            .map(str::to_string);

        let literal_prefix = normalized
            .split('/')
            .take_while(|segment| !segment.contains(['*', '?', '[', '{']))
            .map(unescape_segment)
            .collect();

        Ok(CompiledPattern {
            index,
            body: normalized,
            negated,
            literal_prefix,
            subtree_root,
        })
    }

    /// Whether `relative` (a `/`-separated path relative to the walk root)
    /// matches this pattern under the given case sensitivity.
    fn matches(&self, relative: &str, case: CaseSensitivity) -> bool {
        match case {
            CaseSensitivity::Sensitive => glob_match(&self.body, relative),
            CaseSensitivity::Insensitive => {
                glob_match(self.body.to_ascii_lowercase(), relative.to_ascii_lowercase())
            }
        }
    }

    /// Whether some entry strictly beneath `dir_relative` could match this
    /// pattern. Conservative: may return true for subtrees without matches,
    /// never false for a subtree that contains one.
    fn can_match_below(&self, dir_relative: &str, case: CaseSensitivity) -> bool {
        if self.negated {
            // Negative patterns never justify descending on their own.
            return false;
        }
        let segments: Vec<&str> = dir_relative.split('/').collect();
        if segments.len() < self.literal_prefix.len() {
            // The directory is a (proper) prefix of the literal prefix: it
            // can still lead to a match if every one of its segments equals
            // the corresponding literal segment.
            return segments.iter().zip(&self.literal_prefix).all(|(have, want)| {
                segment_eq(have, want, case)
            });
        }
        // The directory is at or below the end of the literal prefix. The
        // remaining pattern is non-literal, so anything below may match;
        // only the literal segments themselves must agree.
        self.literal_prefix
            .iter()
            .zip(&segments)
            .all(|(want, have)| segment_eq(have, want, case))
    }
}

fn is_glob_special(c: char) -> bool {
    matches!(c, '*' | '?' | '[' | ']' | '{' | '}' | ',' | '!' | '\\')
}

fn unescape_segment(segment: &str) -> String {
    let mut out = String::with_capacity(segment.len());
    let mut chars = segment.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(&next) = chars.peek() {
                out.push(next);
                chars.next();
                continue;
            }
        }
        out.push(c);
    }
    out
}

fn segment_eq(a: &str, b: &str, case: CaseSensitivity) -> bool {
    match case {
        CaseSensitivity::Sensitive => a == b,
        CaseSensitivity::Insensitive => a.eq_ignore_ascii_case(b),
    }
}

/// Configurable directory traversal over compiled glob patterns.
///
/// A `Walker` is a reusable, immutable description of a traversal: patterns
/// and options. Executing it with [`Walker::run`] or [`Walker::explain`]
/// never mutates it (except consuming an installed one-shot [`Fault`]), so
/// the same walker can be rerun after a failure and yields the same result.
#[derive(Debug)]
pub struct Walker {
    patterns: Vec<CompiledPattern>,
    error_policy: ErrorPolicy,
    hidden: HiddenPolicy,
    symlinks: SymlinkPolicy,
    case: CaseSensitivity,
    files_only: bool,
    threads: usize,
    jitter: Duration,
    fault: Option<Fault>,
}

impl Fault {
    /// Creates a fault that makes the next read of `relative_dir` fail once.
    /// Use [`Walker::inject_fault`] to install it.
    #[must_use]
    pub fn once(relative_dir: impl Into<String>) -> Fault {
        Fault {
            relative_dir: relative_dir.into(),
            remaining: Arc::new(AtomicUsize::new(1)),
        }
    }

    /// Whether the fault has already fired.
    #[must_use]
    pub fn fired(&self) -> bool {
        self.remaining.load(AtomicOrdering::Acquire) == 0
    }
}

impl Walker {
    /// Compiles the given positive (`"src/**/*.rs"`) and negative
    /// (`"!target/**"`) patterns. Patterns are validated with [`validate`],
    /// so an invalid pattern is rejected up front through [`WalkError`]'s
    /// existing pattern-error layer.
    pub fn new<I, S>(patterns: I) -> Result<Walker, WalkError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let compiled = patterns
            .into_iter()
            .enumerate()
            .map(|(index, raw)| CompiledPattern::compile(index, raw.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Walker {
            patterns: compiled,
            error_policy: ErrorPolicy::Continue,
            hidden: HiddenPolicy::Skip,
            symlinks: SymlinkPolicy::Skip,
            case: CaseSensitivity::Sensitive,
            files_only: false,
            threads: 1,
            jitter: Duration::ZERO,
            fault: None,
        })
    }

    /// Sets the directory-read error policy.
    #[must_use]
    pub fn error_policy(mut self, policy: ErrorPolicy) -> Walker {
        self.error_policy = policy;
        self
    }

    /// Sets how hidden (dot-prefixed) entries are handled.
    #[must_use]
    pub fn hidden(mut self, policy: HiddenPolicy) -> Walker {
        self.hidden = policy;
        self
    }

    /// Sets symlink handling.
    #[must_use]
    pub fn symlinks(mut self, policy: SymlinkPolicy) -> Walker {
        self.symlinks = policy;
        self
    }

    /// Sets case sensitivity for both pattern literals and entry names.
    #[must_use]
    pub fn case_sensitivity(mut self, case: CaseSensitivity) -> Walker {
        self.case = case;
        self
    }

    /// Only returns regular files; matched directories are still traversed
    /// but excluded from the result set.
    #[must_use]
    pub fn files_only(mut self, yes: bool) -> Walker {
        self.files_only = yes;
        self
    }

    /// Enables parallel directory reading with `n` worker threads. Results
    /// and explain events are sorted deterministically, so the output is
    /// independent of how the workers are scheduled. `n == 1` (the default)
    /// runs the identical engine inline without spawning a thread.
    #[must_use]
    pub fn parallel(mut self, n: usize) -> Walker {
        self.threads = n.max(1);
        self
    }

    /// Inserts a delay before every entry is examined. This does not change
    /// the result; it only perturbs scheduling, useful to prove that
    /// parallel traces are stable under arbitrary interleaving.
    #[must_use]
    pub fn scheduling_jitter(mut self, jitter: Duration) -> Walker {
        self.jitter = jitter;
        self
    }

    /// Installs a one-shot [`Fault`] (replaces any previous fault). The fault
    /// is part of the walker's configuration and is consumed when it fires.
    #[must_use]
    pub fn inject_fault(mut self, fault: Fault) -> Walker {
        self.fault = Some(fault);
        self
    }

    /// The compiled patterns, as they are used by the matcher.
    #[must_use]
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }

    /// Traverses `root` without collecting an explain trace.
    pub fn run(&self, root: impl AsRef<Path>) -> Result<WalkOutput, WalkError> {
        self.execute(root, false)
    }

    /// Traverses `root` and additionally collects a deterministic
    /// [`ExplainTrace`].
    pub fn explain(&self, root: impl AsRef<Path>) -> Result<WalkOutput, WalkError> {
        self.execute(root, true)
    }

    fn execute(&self, root: impl AsRef<Path>, with_trace: bool) -> Result<WalkOutput, WalkError> {
        let root_display = root.as_ref().to_string_lossy().into_owned();
        let canonical_root = match fs::canonicalize(root.as_ref()) {
            Ok(path) => path,
            Err(error) => {
                return Err(WalkError::InvalidRoot {
                    root: root_display,
                    kind: classify_io(&error),
                    message: error.to_string(),
                });
            }
        };
        if !canonical_root.is_dir() {
            return Err(WalkError::InvalidRoot {
                root: root_display,
                kind: IoErrorKind::Other,
                message: "not a directory".to_string(),
            });
        }

        let run_id = RunId::fresh();
        let mut engine = Engine::new(self, root_display, canonical_root, run_id, with_trace);
        engine.drive();

        if let Some(failure) = engine.aborted() {
            return Err(WalkError::Aborted(failure));
        }

        let mut paths = engine.into_paths();
        paths.sort_by(|(left_rel, _), (right_rel, _)| left_rel.cmp(right_rel));

        let mut errors = engine.into_errors();
        errors.sort_by(|left, right| {
            left.relative_dir
                .cmp(&right.relative_dir)
                .then_with(|| left.root.cmp(&right.root))
        });

        let explain_trace = if with_trace {
            let mut events = engine.into_events();
            events.sort_by(|left, right| {
                left.relative
                    .cmp(&right.relative)
                    .then(left.phase.cmp(&right.phase))
                    .then(left.tie.cmp(&right.tie))
            });
            let events = events
                .into_iter()
                .map(|raw| Event {
                    root: root_display.clone(),
                    relative: raw.relative,
                    kind: raw.kind,
                })
                .collect();
            Some(ExplainTrace { run_id, events })
        } else {
            None
        };

        Ok(WalkOutput {
            run_id,
            paths: paths.into_iter().map(|(_, path)| path).collect(),
            errors,
            explain: explain_trace,
        })
    }
}

fn classify_io(error: &std::io::Error) -> IoErrorKind {
    match error.kind() {
        std::io::ErrorKind::NotFound => IoErrorKind::NotFound,
        std::io::ErrorKind::PermissionDenied => IoErrorKind::PermissionDenied,
        _ => IoErrorKind::Other,
    }
}

struct RawEvent {
    relative: String,
    phase: u8,
    tie: u32,
    kind: EventKind,
}

struct Task {
    relative: String,
    absolute: PathBuf,
    /// Canonical directory links already expanded on this branch, used to
    /// prove a followed symlink would close a cycle.
    link_ancestors: Vec<PathBuf>,
}

#[derive(Default)]
struct SharedState {
    pending: Vec<Task>,
    active: usize,
    aborted: Option<IoFailure>,
}

struct Engine<'a> {
    walker: &'a Walker,
    root_display: String,
    canonical_root: PathBuf,
    run_id: RunId,
    with_trace: bool,
    shared: Mutex<SharedState>,
    cv: Condvar,
    stop: std::sync::atomic::AtomicBool,
    results: Mutex<Vec<(String, PathBuf)>>,
    errors: Mutex<Vec<IoFailure>>,
    events: Mutex<Vec<RawEvent>>,
}

const PHASE_COMPILE: u8 = 0;
const PHASE_START: u8 = 1;
const PHASE_PRUNE: u8 = 2;
const PHASE_IGNORE: u8 = 3;
const PHASE_SYMLINK: u8 = 4;
const PHASE_VERDICT: u8 = 5;
const PHASE_READ_FAILED: u8 = 6;

impl<'a> Engine<'a> {
    fn new(
        walker: &'a Walker,
        root_display: String,
        canonical_root: PathBuf,
        run_id: RunId,
        with_trace: bool,
    ) -> Engine<'a> {
        let initial = Task {
            relative: String::new(),
            absolute: canonical_root.clone(),
            link_ancestors: Vec::new(),
        };
        let engine = Engine {
            walker,
            root_display,
            canonical_root,
            run_id,
            with_trace,
            shared: Mutex::new(SharedState {
                pending: vec![initial],
                active: 0,
                aborted: None,
            }),
            cv: Condvar::new(),
            stop: std::sync::atomic::AtomicBool::new(false),
            results: Mutex::new(Vec::new()),
            errors: Mutex::new(Vec::new()),
            events: Mutex::new(Vec::new()),
        };
        if with_trace {
            for pattern in &walker.patterns {
                engine.record(RawEvent {
                    relative: String::new(),
                    phase: PHASE_COMPILE,
                    tie: pattern.index as u32,
                    kind: EventKind::PatternCompiled {
                        index: pattern.index,
                        pattern: pattern.body.clone(),
                        negated: pattern.negated,
                        literal_prefix: pattern.literal_prefix.clone(),
                    },
                });
            }
        }
        engine
    }

    fn aborted(&self) -> Option<IoFailure> {
        self.shared.lock().unwrap().aborted.take()
    }

    fn into_paths(self) -> Vec<(String, PathBuf)> {
        self.results.into_inner().unwrap()
    }

    fn into_errors(self) -> Vec<IoFailure> {
        self.errors.into_inner().unwrap()
    }

    fn into_events(self) -> Vec<RawEvent> {
        self.events.into_inner().unwrap()
    }

    fn record(&self, event: RawEvent) {
        self.events.lock().unwrap().push(event);
    }

    fn drive(&self) {
        if self.walker.threads <= 1 {
            while let Some(task) = self.acquire() {
                if self.stop.load(AtomicOrdering::Acquire) {
                    break;
                }
                self.process_task(task);
            }
            return;
        }

        thread::scope(|scope| {
            for _ in 0..self.walker.threads {
                scope.spawn(|| {
                    while let Some(task) = self.acquire() {
                        if self.stop.load(AtomicOrdering::Acquire) {
                            break;
                        }
                        self.process_task(task);
                    }
                });
            }
        });
    }

    fn acquire(&self) -> Option<Task> {
        let mut state = self.shared.lock().unwrap();
        loop {
            if state.aborted.is_some() {
                return None;
            }
            if let Some(task) = state.pending.pop() {
                state.active += 1;
                return Some(task);
            }
            if state.active == 0 {
                self.cv.notify_all();
                return None;
            }
            state = self.cv.wait(state).unwrap();
        }
    }

    fn release(&self, tasks: Vec<Task>) {
        let mut state = self.shared.lock().unwrap();
        state.active -= 1;
        state.pending.extend(tasks);
        self.cv.notify_all();
    }

    fn process_task(&self, task: Task) {
        let mut discovered = Vec::new();

        if task.relative.is_empty() {
            if self.with_trace {
                self.record(RawEvent {
                    relative: String::new(),
                    phase: PHASE_START,
                    tie: 0,
                    kind: EventKind::StartDir,
                });
            }
        }

        if let Some(failure) = self.read_directory(&task, &mut discovered) {
            self.errors.lock().unwrap().push(failure.clone());
            if self.with_trace {
                self.record(RawEvent {
                    relative: task.relative.clone(),
                    phase: PHASE_READ_FAILED,
                    tie: 0,
                    kind: EventKind::ReadFailed { failure },
                });
            }
            if self.walker.error_policy == ErrorPolicy::Abort {
                let mut state = self.shared.lock().unwrap();
                if state.aborted.is_none() {
                    if let Some(last) = self.errors.lock().unwrap().last() {
                        state.aborted = Some(last.clone());
                    }
                }
                state.active -= 1;
                self.stop.store(true, AtomicOrdering::Release);
                self.cv.notify_all();
                return;
            }
            self.release(Vec::new());
            return;
        }

        // Deterministic order of examination within a directory; this is the
        // order in which child tasks are produced, while the final trace is
        // sorted regardless of when workers consume them.
        discovered.sort_by(|left, right| left.file_name().cmp(&right.file_name()));

        let mut children = Vec::with_capacity(discovered.len());
        for entry in discovered {
            let Some(name) = entry.file_name().to_str().map(str::to_string) else {
                continue;
            };
            let child_relative = if task.relative.is_empty() {
                name.clone()
            } else {
                format!("{}/{}", task.relative, name)
            };
            let child_absolute = task.absolute.join(&name);

            self.jitter();
            self.process_entry(
                &task,
                &name,
                child_relative,
                child_absolute,
                &mut children,
            );
        }
        self.release(children);
    }

    fn jitter(&self) {
        if self.walker.jitter != Duration::ZERO {
            thread::sleep(self.walker.jitter);
        }
    }

    /// Reads a directory after consuming an installed fault matching it.
    fn read_directory(&self, task: &Task, discovered: &mut Vec<DirEntry>) -> Option<IoFailure> {
        if let Some(fault) = &self.walker.fault {
            if fault.relative_dir == task.relative
                && fault
                    .remaining
                    .fetch_update(
                        AtomicOrdering::AcqRel,
                        AtomicOrdering::Acquire,
                        |count| if count > 0 { Some(count - 1) } else { None },
                    )
                    .is_ok()
            {
                return Some(IoFailure {
                    relative_dir: task.relative.clone(),
                    root: self.root_display.clone(),
                    kind: IoErrorKind::Injected,
                    message: "injected fault".to_string(),
                });
            }
        }

        let read: ReadDir = match fs::read_dir(&task.absolute) {
            Ok(read) => read,
            Err(error) => {
                return Some(IoFailure {
                    relative_dir: task.relative.clone(),
                    root: self.root_display.clone(),
                    kind: classify_io(&error),
                    message: error.to_string(),
                });
            }
        };

        for entry in read {
            match entry {
                Ok(entry) => discovered.push(entry),
                Err(error) => {
                    return Some(IoFailure {
                        relative_dir: task.relative.clone(),
                        root: self.root_display.clone(),
                        kind: classify_io(&error),
                        message: error.to_string(),
                    });
                }
            }
        }
        None
    }
