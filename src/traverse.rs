//! Deterministic directory traversal with an optional `explain` trail.
//!
//! This module layers a real filesystem walker on top of the crate's single
//! matching engine ([`crate::glob_match`]). The default entry, [`walk`], keeps
//! the lean "matcher only" profile of the crate; [`explain`] runs the exact
//! same traversal and additionally records a deterministic trail describing
//! why every encountered path was included or excluded:
//!
//! - pattern compilation ([`ExplainEvent::PatternCompiled`]),
//! - the chosen start directory ([`ExplainEvent::StartDir`]),
//! - directory pruning ([`ExplainEvent::Pruned`]),
//! - ignore hits from negated patterns ([`ExplainEvent::Ignored`]),
//! - symlink decisions ([`ExplainEvent::SymlinkDecision`]),
//! - and the provenance of every final path ([`ExplainEvent::Included`]).
//!
//! Both entries share one traversal implementation; recording is a compile-time
//! sink choice, so `walk` performs no event bookkeeping.
//!
//! # Result determinism
//!
//! Results and the explain trail are canonical: paths are reported with `/`
//! separators, directory entries are visited in sorted order, and the parallel
//! scheduler merges worker output back into the same canonical order. Thread
//! scheduling therefore never changes the order of results or events.

use std::collections::HashSet;
use std::ffi::OsString;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use crate::{glob_match, validate, Error as PatternError};

/// Identity of a compiled pattern set. Every [`Matcher`] gets a unique value;
/// nothing produced by one matcher is ever reused as if it came from another.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MatcherId(u64);

/// Identity of a single traversal run. Every call to [`walk`]/[`explain`] gets
/// a fresh value, and every event in the resulting trail carries that same
/// value, so trails cannot be confused with state from a previous (failed) run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RunId(u64);

static MATCHER_COUNTER: AtomicU64 = AtomicU64::new(1);
static RUN_COUNTER: AtomicU64 = AtomicU64::new(1);

fn next_matcher_id() -> MatcherId {
    MatcherId(MATCHER_COUNTER.fetch_add(1, Ordering::Relaxed))
}

fn next_run_id() -> RunId {
    RunId(RUN_COUNTER.fetch_add(1, Ordering::Relaxed))
}

/// What the walker should do when reading a directory fails (e.g. permission
/// denied).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorPolicy {
    /// Stop and return the first error as a [`WalkError`].
    #[default]
    Return,
    /// Record the error in the explain trail (or skip it silently in
    /// [`walk`]) and continue traversing everything else.
    Skip,
}

/// Options for [`walk`], [`explain`] and [`Matcher::walk`]/[`Matcher::explain`].
///
/// `Options` implements [`Default`]; the defaults match the crate's matcher
/// semantics: case-sensitive matching, no symlink following, hidden entries
/// are only visible when a pattern explicitly names a dot segment.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Options {
    /// Match paths case-insensitively. Defaults to `false`, matching the
    /// byte-for-byte semantics of [`crate::glob_match`].
    pub case_insensitive: bool,
    /// Follow symbolic links to directories. Defaults to `false`. Even when
    /// enabled, links that would form a cycle are never followed.
    pub follow_symlinks: bool,
    /// Include hidden (dot) entries even when no pattern names a dot segment.
    /// Defaults to `false`.
    pub include_hidden: bool,
    /// Behavior when a directory cannot be read. Defaults to
    /// [`ErrorPolicy::Return`].
    pub on_error: ErrorPolicy,
    /// Number of worker threads. `None` selects a bounded default based on the
    /// available parallelism; `Some(1)` runs inline without spawning threads.
    pub workers: Option<usize>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            case_insensitive: false,
            follow_symlinks: false,
            include_hidden: false,
            on_error: ErrorPolicy::Return,
            workers: None,
        }
    }
}

impl Options {
    /// Creates options with the default behavior.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enables or disables case-insensitive matching.
    pub fn case_insensitive(mut self, yes: bool) -> Self {
        self.case_insensitive = yes;
        self
    }

    /// Enables or disables following symbolic links to directories.
    pub fn follow_symlinks(mut self, yes: bool) -> Self {
        self.follow_symlinks = yes;
        self
    }

    /// Enables or disables including hidden entries unconditionally.
    pub fn include_hidden(mut self, yes: bool) -> Self {
        self.include_hidden = yes;
        self
    }

    /// Sets the directory-read error policy.
    pub fn on_error(mut self, policy: ErrorPolicy) -> Self {
        self.on_error = policy;
        self
    }

    /// Pins the number of worker threads. `1` runs inline.
    pub fn workers(mut self, workers: usize) -> Self {
        self.workers = Some(workers.max(1));
        self
    }

    fn resolved_workers(&self) -> usize {
        match self.workers {
            Some(n) => n.max(1),
            None => thread::available_parallelity().map(usize::from).unwrap_or(1).min(8),
        }
    }
}

/// Why a directory subtree was pruned without descending into it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PruneReason {
    /// No positive pattern can match anything beneath the directory (literal
    /// prefix or fixed-depth mismatch).
    Unreachable,
    /// A negated pattern with a literal subtree prefix ignores the whole
    /// subtree.
    IgnorePattern,
}

/// Why a leaf entry that was not included was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExcludeReason {
    /// The entry is hidden and no pattern explicitly names a dot segment.
    Hidden,
    /// No positive pattern matches the entry.
    NoPositiveMatch,
    /// A negated pattern matches the entry.
    IgnorePattern,
    /// A dangling symbolic link whose target does not exist.
    BrokenSymlink,
}

/// The decision recorded for a symbolic link.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SymlinkAction {
    /// Link following is disabled; the link is treated as a leaf entry.
    NotFollowed,
    /// The link resolves to a directory and was descended into.
    FollowedDirectory,
    /// The link resolves to a file and is treated as a leaf entry.
    FollowedFile,
    /// Following the link would revisit an ancestor directory, so it was cut.
    CycleDetected,
}

/// Why a directory read failed.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ReadError {
    /// Run during which the read failed.
    pub run_id: RunId,
    /// The directory that could not be read, relative to the walk root with
    /// `/` separators (empty for the walk root itself).
    pub path: String,
    /// The operating system error message.
    pub message: String,
    /// The raw OS error number, when available.
    pub raw_os_error: Option<i32>,
}

/// One step of the deterministic explain trail.
///
/// Events are grouped by their canonical path: compilation first, then the
/// start directory, then per-path events in sorted path order.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExplainEvent {
    /// A pattern was parsed into the segments used for traversal decisions.
    PatternCompiled {
        /// Run this event belongs to.
        run_id: RunId,
        /// Zero-based position in the supplied pattern list.
        index: usize,
        /// The pattern exactly as supplied.
        pattern: String,
        /// Whether the pattern begins with one or more `!`.
        negated: bool,
        /// Path segments split on unescaped `/`.
        segments: Vec<String>,
        /// Whether any segment explicitly begins with `.`.
        allows_hidden: bool,
    },
    /// The static prefix common to every positive pattern was resolved to a
    /// concrete starting directory (or found not to exist).
    StartDir {
        /// Run this event belongs to.
        run_id: RunId,
        /// The requested walk root.
        root: String,
        /// Static prefix segments consumed while descending from `root`.
        prefix: Vec<String>,
        /// Directory the traversal actually starts at.
        start: String,
        /// Whether `start` exists on disk.
        exists: bool,
    },
    /// A directory was not descended into.
    Pruned {
        /// Run this event belongs to.
        run_id: RunId,
        /// Canonical path of the directory.
        path: String,
        /// Why traversal stopped there.
        reason: PruneReason,
        /// Index of the negated pattern responsible, when
        /// `reason == PruneReason::IgnorePattern`.
        pattern_index: Option<usize>,
    },
    /// A reachable entry was excluded by a negated pattern.
    Ignored {
        /// Run this event belongs to.
        run_id: RunId,
        /// Canonical path of the entry.
        path: String,
        /// Index of the negated pattern that matched.
        pattern_index: usize,
        /// Whether the ignored entry was a directory whose subtree was cut.
        is_dir: bool,
    },
    /// A symbolic link was encountered and classified.
    SymlinkDecision {
        /// Run this event belongs to.
        run_id: RunId,
        /// Canonical path of the link.
        path: String,
        /// What the walker did with the link.
        action: SymlinkAction,
    },
    /// A directory could not be read.
    ReadError(ReadError),
    /// A leaf entry was evaluated against the patterns and rejected.
    Excluded {
        /// Run this event belongs to.
        run_id: RunId,
        /// Canonical path of the entry.
        path: String,
        /// Why it was rejected.
        reason: ExcludeReason,
    },
    /// A leaf entry matched and appears in the final result set.
    Included {
        /// Run this event belongs to.
        run_id: RunId,
        /// Canonical path of the entry, relative to the walk root.
        path: String,
        /// Index of the positive pattern that matched.
        pattern_index: usize,
    },
}

impl ExplainEvent {
    /// Run that produced this event.
    pub fn run_id(&self) -> RunId {
        match self {
            ExplainEvent::PatternCompiled { run_id, .. }
            | ExplainEvent::StartDir { run_id, .. }
            | ExplainEvent::Pruned { run_id, .. }
            | ExplainEvent::Ignored { run_id, .. }
            | ExplainEvent::SymlinkDecision { run_id, .. }
            | ExplainEvent::Excluded { run_id, .. }
            | ExplainEvent::Included { run_id, .. } => *run_id,
            ExplainEvent::ReadError(e) => e.run_id,
        }
    }
}

impl fmt::Display for ExplainEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExplainEvent::PatternCompiled {
                index,
                pattern,
                negated,
                segments,
                ..
            } => write!(
                f,
                "compile[{index}]: {pattern:?} (negated={negated}, segments={segments:?})"
            ),
            ExplainEvent::StartDir {
                prefix, start, exists, ..
            } => write!(
                f,
                "start: prefix={prefix:?} -> {start:?} (exists={exists})"
            ),
            ExplainEvent::Pruned {
                path,
                reason,
                pattern_index,
                ..
            } => write!(f, "prune: {path:?} reason={reason:?} by={pattern_index:?}"),
            ExplainEvent::Ignored {
                path,
                pattern_index,
                is_dir,
                ..
            } => write!(
                f,
                "ignore: {path:?} matched negated pattern {pattern_index} (dir={is_dir})"
            ),
            ExplainEvent::SymlinkDecision { path, action, .. } => {
                write!(f, "symlink: {path:?} -> {action:?}")
            }
            ExplainEvent::ReadError(e) => {
                write!(f, "read-error: {:?}: {}", e.path, e.message)
            }
            ExplainEvent::Excluded { path, reason, .. } => {
                write!(f, "exclude: {path:?} reason={reason:?}")
            }
            ExplainEvent::Included {
                path, pattern_index, ..
            } => write!(f, "include: {path:?} via positive pattern {pattern_index}"),
        }
    }
}
