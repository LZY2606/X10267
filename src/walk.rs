//! Directory walking on top of [`glob_match`], with an optional, deterministic
//! explain trace.
//!
//! [`glob_walk`] is an additive entry point: it reuses the existing matcher
//! ([`glob_match`]) and validator ([`validate`]) instead of reimplementing
//! them, so case, separator and negation semantics are exactly those of the
//! matching API. The default [`WalkOptions`] add no tracing overhead — the
//! explain trace is only collected when [`WalkOptions::explain`] is set.
//!
//! Every call is a self-contained session with a unique, monotonically
//! increasing [`WalkOutcome::session`] identity. No cache, counter or
//! traversal state is shared between calls, so a failed call cannot leak
//! partial state into a later one: retrying with identical input and options
//! yields identical results (only the session identity differs).

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{Error, glob_match, validate};

/// Session identity source. Each `glob_walk` call takes one unique, increasing
/// value; it is only ever read-and-incremented, never written back, so no
/// per-call state can leak through it.
static NEXT_SESSION: AtomicU64 = AtomicU64::new(1);

/// Options controlling a [`glob_walk`] call.
///
/// The defaults match the behavior of the existing matching entry points:
/// case-sensitive matching, `/` (and the platform separator) treated as path
/// separators, and no tracing.
#[derive(Debug, Clone)]
pub struct WalkOptions {
    /// When `false`, patterns and candidate paths are compared
    /// case-insensitively (Unicode lowercase). Defaults to `true`, matching
    /// [`glob_match`].
    pub case_sensitive: bool,
    /// When `false` (the default), files and directories whose name starts
    /// with `.` are skipped; pruned directories are reported in the trace.
    pub dot: bool,
    /// When `true` (the default), symbolic links to directories are followed.
    /// A link that resolves to a directory already on the current ancestor
    /// chain forms a loop and is never followed, regardless of this flag.
    pub follow_symlinks: bool,
    /// How I/O errors (e.g. permission denied on a directory) are presented.
    /// Defaults to [`ErrorPolicy::Skip`].
    pub error_policy: ErrorPolicy,
    /// When `true`, collect an [`ExplainTrace`] describing every traversal
    /// decision. The trace never changes the result set. Defaults to `false`.
    pub explain: bool,
}

impl Default for WalkOptions {
    fn default() -> Self {
        Self {
            case_sensitive: true,
            dot: false,
            follow_symlinks: true,
            error_policy: ErrorPolicy::Skip,
            explain: false,
        }
    }
}

/// How [`glob_walk`] presents I/O errors encountered during traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ErrorPolicy {
    /// Record the error in [`WalkOutcome::errors`] (and in the trace) and
    /// continue with the remaining entries. This is the default.
    #[default]
    Skip,
    /// Abort the walk and return the first error from [`glob_walk`]. No
    /// partial result is returned and no state is kept, so the call can be
    /// retried once the cause is fixed.
    Fail,
}

/// An error produced or collected by [`glob_walk`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WalkError {
    /// A pattern was rejected by the existing [`validate`] entry point; the
    /// original [`Error`] is preserved so callers see the same error layering
    /// as the matching API.
    InvalidPattern(Error),
    /// An I/O error while reading or resolving `path`.
    Io {
        /// The directory or entry that could not be read.
        path: PathBuf,
        /// The kind of the underlying I/O error.
        kind: io::ErrorKind,
        /// The underlying error message.
        message: String,
    },
}

impl fmt::Display for WalkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WalkError::InvalidPattern(error) => write!(f, "invalid pattern: {error}"),
            WalkError::Io { path, message, .. } => {
                write!(f, "I/O error at {}: {message}", path.display())
            }
        }
    }
}

impl std::error::Error for WalkError {}

/// Why a directory was pruned (not descended into).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruneReason {
    /// The directory name starts with `.` and [`WalkOptions::dot`] is off.
    Hidden,
    /// The directory itself matched a negative (`!`) pattern.
    Ignored,
}

/// What was decided for a symbolic link to a directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymlinkDecision {
    /// The link was followed and its target traversed.
    Followed,
    /// The link resolves to a directory already on the ancestor chain; it was
    /// not followed, so traversal always terminates.
    SkippedLoop,
    /// [`WalkOptions::follow_symlinks`] is off; the link was not followed.
    NotFollowed,
}

/// A single, ordered explain-trace event.
///
/// Events are emitted in a deterministic order: directory entries are sorted
/// by name before visiting, so the trace does not depend on filesystem
/// enumeration order or thread scheduling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceEvent {
    /// A pattern was validated and compiled (split into positive/negative).
    PatternCompiled {
        /// Index of the pattern in the input slice.
        index: usize,
        /// The pattern as provided by the caller.
        pattern: String,
        /// Whether the pattern is negative (odd number of leading `!`).
        negative: bool,
    },
    /// Traversal started at this directory.
    StartDirectory {
        /// The root directory passed to [`glob_walk`].
        path: PathBuf,
    },
    /// A directory was not descended into.
    DirPruned {
        /// The pruned directory.
        path: PathBuf,
        /// Why it was pruned.
        reason: PruneReason,
    },
    /// A file was excluded because it matched a negative pattern.
    IgnoreHit {
        /// The excluded file.
        path: PathBuf,
        /// Index of the negative pattern that matched.
        pattern: usize,
    },
    /// A decision was made for a symbolic link to a directory.
    SymlinkDecision {
        /// The symbolic link.
        path: PathBuf,
        /// The decision that was taken.
        decision: SymlinkDecision,
    },
    /// A file matched a positive pattern and is part of the result; this is
    /// the provenance of every path in [`WalkOutcome::paths`].
    EntryMatched {
        /// The included file.
        path: PathBuf,
        /// Index of the (first) positive pattern that matched.
        pattern: usize,
    },
    /// An I/O error was recorded (see [`WalkOptions::error_policy`]).
    Error {
        /// The directory or entry that could not be read.
        path: PathBuf,
        /// The underlying error message.
        message: String,
    },
}

/// The deterministic explain trace collected when [`WalkOptions::explain`] is
/// set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExplainTrace {
    /// Identity of the call that produced this trace; see
    /// [`WalkOutcome::session`].
    pub session: u64,
    /// The events, in traversal order.
    pub events: Vec<TraceEvent>,
}

/// The result of a successful [`glob_walk`] call.
#[derive(Debug, Clone)]
pub struct WalkOutcome {
    /// Unique, monotonically increasing identity of this call. It versions
    /// everything the call produced: two calls never share a session, and no
    /// state from a previous (possibly failed) call is reused.
    pub session: u64,
    /// Matching files, sorted and deduplicated. Each path is
    /// `root`-relative-joined (i.e. under the root passed to [`glob_walk`]).
    pub paths: Vec<PathBuf>,
    /// I/O errors collected under [`ErrorPolicy::Skip`].
    pub errors: Vec<WalkError>,
    /// The explain trace, present iff [`WalkOptions::explain`] was set.
    pub trace: Option<ExplainTrace>,
}

/// Walks `root` recursively, returning files matching `patterns` against
/// their `/`-separated path relative to `root`.
///
/// Patterns follow the existing crate syntax; matching itself is delegated to
/// [`glob_match`], and each pattern is checked up front with [`validate`] —
/// an invalid pattern is rejected as [`WalkError::InvalidPattern`] before any
/// traversal happens. A pattern with an odd number of leading `!` is negative:
/// a file is included iff it matches at least one positive pattern and no
/// negative pattern. A directory matched by a negative pattern is pruned
/// entirely.
///
/// Only files (including symbolic links to files) are returned, never
/// directories. The result is sorted, deduplicated and independent of
/// filesystem enumeration order and thread scheduling.
///
/// ```rust
/// use fast_glob::{WalkOptions, glob_walk};
///
/// let outcome = glob_walk(".", &["src/**/*.rs"], &WalkOptions::default()).unwrap();
/// assert!(outcome.paths.iter().any(|p| p.ends_with("lib.rs")));
/// ```
pub fn glob_walk(
    root: impl AsRef<Path>,
    patterns: &[impl AsRef<str>],
    options: &WalkOptions,
) -> Result<WalkOutcome, WalkError> {
    let session = NEXT_SESSION.fetch_add(1, Ordering::Relaxed);
    let root = root.as_ref();

    let mut events = options.explain.then(Vec::new);
    let mut positives = Vec::new();
    let mut negatives = Vec::new();
    for (index, raw) in patterns.iter().enumerate() {
        let raw = raw.as_ref();
        // Reject invalid patterns through the existing validation entry point
        // before touching the filesystem, so a rejected call is side-effect
        // free and repeatable.
        validate(raw).map_err(WalkError::InvalidPattern)?;

        let bangs = raw.bytes().take_while(|&b| b == b'!').count();
        let negative = bangs % 2 == 1;
        let mut body = raw[bangs..].to_string();
        if !options.case_sensitive {
            body = body.to_lowercase();
        }
        if let Some(events) = &mut events {
            events.push(TraceEvent::PatternCompiled {
                index,
                pattern: raw.to_string(),
                negative,
            });
        }
        (if negative { &mut negatives } else { &mut positives })
            .push(CompiledPattern { index, body });
    }

    if let Some(events) = &mut events {
        events.push(TraceEvent::StartDirectory { path: root.to_path_buf() });
    }

    let mut walker = Walker {
        options,
        positives,
        negatives,
        paths: Vec::new(),
        errors: Vec::new(),
        events,
        ancestors: vec![fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf())],
    };
    walker.walk_dir(root, "")?;

    let mut paths = walker.paths;
    paths.sort();
    paths.dedup();

    Ok(WalkOutcome {
        session,
        paths,
        errors: walker.errors,
        trace: walker.events.map(|events| ExplainTrace { session, events }),
    })
}

struct CompiledPattern {
    index: usize,
    body: String,
}

struct Walker<'a> {
    options: &'a WalkOptions,
    positives: Vec<CompiledPattern>,
    negatives: Vec<CompiledPattern>,
    paths: Vec<PathBuf>,
    errors: Vec<WalkError>,
    events: Option<Vec<TraceEvent>>,
    /// Canonical paths of the directories on the current descent stack, used
    /// to detect (and refuse) symlink loops.
    ancestors: Vec<PathBuf>,
}

impl Walker<'a> {
    fn event(&mut self, event: TraceEvent) {
        if let Some(events) = &mut self.events {
            events.push(event);
        }
    }

    fn record_io_error(&mut self, path: &Path, error: io::Error) -> Result<(), WalkError> {
        self.event(TraceEvent::Error {
            path: path.to_path_buf(),
            message: error.to_string(),
        });
        let error = WalkError::Io {
            path: path.to_path_buf(),
            kind: error.kind(),
            message: error.to_string(),
        };
        match self.options.error_policy {
            ErrorPolicy::Fail => Err(error),
            ErrorPolicy::Skip => {
                self.errors.push(error);
                Ok(())
            }
        }
    }

    fn matches(&self, pattern: &CompiledPattern, candidate: &str) -> bool {
        glob_match(&pattern.body, candidate)
    }

    fn candidate(&self, rel: &str) -> String {
        if self.options.case_sensitive { rel.to_string() } else { rel.to_lowercase() }
    }

    fn walk_dir(&mut self, dir: &Path, rel: &str) -> Result<(), WalkError> {
        let entries = match fs::read_dir(dir) {
            Ok(entries) => entries.collect::<Result<Vec<_>, _>>(),
            Err(error) => return self.record_io_error(dir, error),
        };
        let mut entries = match entries {
            Ok(entries) => entries,
            Err(error) => return self.record_io_error(dir, error),
        };
        // Sort by name so results and traces do not depend on enumeration order.
        entries.sort_by_key(fs::DirEntry::file_name);

        for entry in entries {
            self.visit(&entry, rel)?;
        }
        Ok(())
    }

    fn visit(&mut self, entry: &fs::DirEntry, rel: &str) -> Result<(), WalkError> {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let rel_child = if rel.is_empty() { name.to_string() } else { format!("{rel}/{name}") };

        let file_type = match entry.file_type() {
            Ok(file_type) => file_type,
            Err(error) => return self.record_io_error(&path, error),
        };

        let (is_dir, via_symlink) = if file_type.is_symlink() {
            match fs::metadata(&path) {
                Ok(metadata) => (metadata.is_dir(), true),
                // A dangling link cannot be descended into; treat it as a file.
                Err(_) => (false, true),
            }
        } else {
            (file_type.is_dir(), false)
        };

        if !self.options.dot && name.starts_with('.') {
            if is_dir {
                self.event(TraceEvent::DirPruned { path, reason: PruneReason::Hidden });
            }
            return Ok(());
        }

        if !is_dir {
            return self.visit_file(&path, &rel_child);
        }

        if via_symlink {
            return self.visit_symlinked_dir(&path, &rel_child);
        }

        let candidate = self.candidate(&rel_child);
        if let Some(pattern) =
            self.negatives.iter().find(|pattern| self.matches(pattern, &candidate))
        {
            self.event(TraceEvent::DirPruned {
                path,
                reason: PruneReason::Ignored,
            });
            let _ = pattern;
            return Ok(());
        }

        self.descend(&path, &rel_child)
    }

    fn visit_symlinked_dir(&mut self, path: &Path, rel_child: &str) -> Result<(), WalkError> {
        if !self.options.follow_symlinks {
            self.event(TraceEvent::SymlinkDecision {
                path: path.to_path_buf(),
                decision: SymlinkDecision::NotFollowed,
            });
            return Ok(());
        }

        let canonical = match fs::canonicalize(path) {
            Ok(canonical) => canonical,
            Err(error) => return self.record_io_error(path, error),
        };
        if self.ancestors.contains(&canonical) {
            self.event(TraceEvent::SymlinkDecision {
                path: path.to_path_buf(),
                decision: SymlinkDecision::SkippedLoop,
            });
            return Ok(());
        }

        self.event(TraceEvent::SymlinkDecision {
            path: path.to_path_buf(),
            decision: SymlinkDecision::Followed,
        });
        self.descend(path, rel_child)
    }

    fn descend(&mut self, dir: &Path, rel: &str) -> Result<(), WalkError> {
        let canonical = match fs::canonicalize(dir) {
            Ok(canonical) => canonical,
            Err(error) => return self.record_io_error(dir, error),
        };
        self.ancestors.push(canonical);
        let result = self.walk_dir(dir, rel);
        self.ancestors.pop();
        result
    }

    fn visit_file(&mut self, path: &Path, rel_child: &str) -> Result<(), WalkError> {
        let candidate = self.candidate(rel_child);
        if let Some(pattern) =
            self.negatives.iter().find(|pattern| self.matches(pattern, &candidate))
        {
            self.event(TraceEvent::IgnoreHit {
                path: path.to_path_buf(),
                pattern: pattern.index,
            });
            return Ok(());
        }
        if let Some(pattern) =
            self.positives.iter().find(|pattern| self.matches(pattern, &candidate))
        {
            self.event(TraceEvent::EntryMatched {
                path: path.to_path_buf(),
                pattern: pattern.index,
            });
            self.paths.push(path.to_path_buf());
        }
        Ok(())
    }
}
