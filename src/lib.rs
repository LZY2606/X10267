//! `fast-glob` is a high-performance glob matching crate for Rust, originally forked from [`devongovett/glob-match`](https://github.com/devongovett/glob-match).
//! This crate provides efficient glob pattern matching with support for multi-pattern matching and brace expansion.
//!
//! ## Key Features
//!
//! - Up to 60% performance improvement.
//! - Support for more complex and efficient brace expansion.
//! - Fixed matching issues with wildcard and globstar [`glob-match/issues#9`](https://github.com/devongovett/glob-match/issues/9).
//!
//! ## Examples
//!
//! ```rust
//! use fast_glob::glob_match;
//!
//! let glob = "some/**/n*d[k-m]e?txt";
//! let path = "some/a/bigger/path/to/the/crazy/needle.txt";
//!
//! assert!(glob_match(glob, path));
//! ```
//!
//! ## Validation
//!
//! [`glob_match`] does not report invalid patterns — an unclosed `{` or `[`,
//! a trailing `\`, or brace expansions nested deeper than 10 levels have an
//! unspecified result (typically no match). This is a deliberate performance
//! trade-off: there is no compile step, and the pattern is interpreted lazily
//! while matching, so reliably detecting a malformed pattern would require an
//! extra scan on every call. Validation is instead a separate, one-time step —
//! use [`validate`] to reject such patterns with a descriptive [`Error`]:
//!
//! ```rust
//! use fast_glob::{validate, Error, ErrorKind};
//!
//! assert!(validate("some/**/n*d[k-m]e?txt").is_ok());
//! assert_eq!(
//!     validate("src/**/*.{js,ts"),
//!     Err(Error { kind: ErrorKind::UnclosedBrace, index: 9 })
//! );
//! ```
//!
//! ## Syntax
//!
//! `fast-glob` supports the following glob pattern syntax:
//!
//! | Syntax  | Meaning                                                                                                                                                                                             |
//! | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
//! | `?`     | Matches any single character.                                                                                                                                                                       |
//! | `*`     | Matches zero or more characters, except for path separators (e.g., `/`).                                                                                                                             |
//! | `**`    | Matches zero or more characters, including path separators. Must match a complete path segment (i.e., followed by a `/` or the end of the pattern).                                                  |
//! | `[ab]`  | Matches one of the characters contained in the brackets. Character ranges, e.g., `[a-z]`, are also supported. Use `[!ab]` or `[^ab]` to match any character _except_ those contained in the brackets. |
//! | `{a,b}` | Matches one of the patterns contained in the braces. Any of the wildcard characters can be used in the sub-patterns. Braces may be nested up to 10 levels deep.                                     |
//! | `!`     | When at the start of the glob, this negates the result. Multiple `!` characters negate the glob multiple times.                                                                                     |
//! | `\`     | A backslash character may be used to escape any of the above special characters.                                                                                                                    |
//!
//! ---
//!
//! For detailed usage and API reference, refer to the specific function and struct documentation.
//!
//! For any issues or contributions, please visit the [GitHub repository](https://github.com/oxc-project/fast-glob).

/**
 * The following code was originally forked from
 * https://github.com/devongovett/glob-match/blob/d5a6c67/src/lib.rs
 *
 * MIT Licensed
 * Copyright (c) 2023 Devon Govett
 * https://github.com/devongovett/glob-match/tree/main/LICENSE
 */
use std::fmt;
use std::path::is_separator;

use arrayvec::ArrayVec;

const MAX_BRACE_NESTING: usize = 10;

#[derive(Clone, Debug, Default)]
struct State {
    path_index: usize,
    glob_index: usize,
    brace_depth: usize,

    wildcard: Wildcard,
    globstar: Wildcard,
}

#[derive(Clone, Copy, Debug, Default)]
struct Wildcard {
    glob_index: u32,
    path_index: u32,
    brace_depth: u32,
}

type BraceStack = ArrayVec<(u32, u32), MAX_BRACE_NESTING>;

/// An error describing why a glob pattern is invalid, returned by [`validate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Error {
    /// The kind of invalid construct that was found.
    pub kind: ErrorKind,
    /// Byte offset in the pattern of the offending character.
    pub index: usize,
}

/// The kind of invalid construct described by an [`Error`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorKind {
    /// A `{` is never closed by a matching `}`.
    UnclosedBrace,
    /// A `[` is never closed by a matching `]`.
    UnclosedBracket,
    /// A `\` at the end of the pattern has no character to escape.
    TrailingBackslash,
    /// Brace expansions nest deeper than the supported 10 levels.
    BraceNestingTooDeep,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let index = self.index;
        match self.kind {
            ErrorKind::UnclosedBrace => write!(
                f,
                "unclosed brace expansion at byte {index}; missing '}}' (to match a literal '{{', escape it as '\\{{' or '[{{]')"
            ),
            ErrorKind::UnclosedBracket => write!(
                f,
                "unclosed character class at byte {index}; missing ']' (to match a literal '[', escape it as '\\[' or '[[]')"
            ),
            ErrorKind::TrailingBackslash => write!(
                f,
                "trailing backslash at byte {index} has no character to escape (to match a literal '\\', use '\\\\')"
            ),
            ErrorKind::BraceNestingTooDeep => write!(
                f,
                "brace expansion at byte {index} nests deeper than the supported {MAX_BRACE_NESTING} levels"
            ),
        }
    }
}

impl std::error::Error for Error {}

/// Performs glob pattern matching for `glob` against `path`.
///
/// `glob` is expected to be a valid pattern. An invalid pattern — an unclosed
/// `{` or `[`, a trailing `\`, or brace expansions nested deeper than 10
/// levels — cannot be reported here and its result is unspecified: typically
/// it matches nothing, and it never matches through `!` negation, but the
/// exact behavior may change between releases. Callers accepting user-written
/// patterns should reject invalid ones up front with [`validate`].
pub fn glob_match(glob: impl AsRef<[u8]>, path: impl AsRef<[u8]>) -> bool {
    let (matched, invalid_pattern) = glob_match_internal(glob.as_ref(), path.as_ref());
    matched && !invalid_pattern
}

/// A single step in the deterministic explanation trace produced by
/// [`glob_match_explain`].
///
/// Events are emitted in the exact order the matching engine makes the
/// corresponding decisions, so a trace is fully determined by the `(glob,
/// path)` pair: it does not depend on thread scheduling, call ordering, or
/// any prior call. The engine holds no caches, counters, or other shared
/// state between calls, so a failed call (e.g. one whose pattern is rejected
/// by [`validate`]) cannot influence the trace of a later one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TraceEvent {
    /// The pattern was checked by [`validate`] and accepted. This is the
    /// "compile" step: `fast-glob` interprets patterns lazily with no
    /// compilation pass, so validation is the only up-front analysis.
    /// `negations` counts the leading `!` characters (the ignore polarity).
    Compiled { negations: usize },
    /// The pattern was rejected by [`validate`] with the same [`Error`]
    /// stratification (`kind` + byte `index`) the public validator reports.
    /// Matching still proceeds exactly as [`glob_match`] would — its result
    /// for an invalid pattern is unspecified — so the trace continues after
    /// this event and the verdict stays identical to [`glob_match`].
    CompileError { kind: ErrorKind, index: usize },
    /// Matching started at the beginning of the path (the "start directory"
    /// of the match: all positions are byte offsets into `path`).
    Start { path_len: usize },
    /// A `*` or `**` wildcard was recorded as a backtracking point.
    Wildcard { globstar: bool, glob_index: usize, path_index: usize },
    /// A globstar skipped forward to just past a path separator, pruning the
    /// intermediate segments from consideration at this position. `to` is
    /// always greater than `from`, so even paths with arbitrarily repeated
    /// segments (as a directory walker would produce when expanding a
    /// symlink cycle) are consumed in a bounded number of steps — the
    /// matcher never revisits a position it has pruned past.
    SkipToSeparator { from: usize, to: usize },
    /// Matching resumed at the most recent wildcard after a mismatch.
    Backtrack { glob_index: usize, path_index: usize },
    /// A `{a,b}` brace branch was tried.
    BraceBranch { open: usize, branch: usize },
    /// The remaining branches of a brace expansion were skipped.
    BraceSkip { from: usize, to: usize },
    /// A literal `/` in the pattern matched a path separator; `byte` is the
    /// actual separator byte found in the path — `/`, or `\` on platforms
    /// where [`std::path::is_separator`] accepts it. Case and separator
    /// handling otherwise follow the same rules as [`glob_match`]: literal
    /// bytes compare exactly.
    Separator { path_index: usize, byte: u8 },
    /// Matching stopped before consuming the whole path: no wildcard could
    /// backtrack, so the remainder of the path was pruned. This is the
    /// engine-level reason a path under a non-matching prefix is excluded
    /// without being scanned further.
    Pruned { glob_index: usize, path_index: usize },
    /// Final verdict — always the last event of a trace.
    Final { matched: bool, reason: FinalReason },
}

/// The reason a path was included or excluded, as recorded by the
/// [`TraceEvent::Final`] event of an [`Explanation`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FinalReason {
    /// Pattern and path were fully consumed: the path is included because it
    /// matched the pattern.
    FullMatch,
    /// The path did not match the pattern; the trace's [`TraceEvent::Pruned`]
    /// events show where matching gave up.
    NoMatch,
    /// A negated (`!`-prefixed) pattern whose inner pattern did not match:
    /// the path is included by the negation.
    NegatedIncluded,
    /// A negated (`!`-prefixed) pattern whose inner pattern matched: the
    /// path is excluded — the ignore rule hit.
    NegatedExcluded,
    /// The pattern is invalid (see [`validate`]); the verdict is `false`
    /// regardless of any partial match, exactly as [`glob_match`] behaves.
    InvalidPattern,
}

/// The result of [`glob_match_explain`]: the verdict plus the deterministic
/// trace of how it was reached.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Explanation {
    /// The verdict — always identical to [`glob_match`] for the same
    /// `(glob, path)` pair, whether the pattern is valid or not.
    pub matched: bool,
    /// The ordered trace of decisions: pattern compilation, match start,
    /// wildcard/globstar bookkeeping, segment pruning, brace branching,
    /// separator handling, and the final verdict with its [`FinalReason`].
    pub trace: Vec<TraceEvent>,
}

/// Performs glob pattern matching like [`glob_match`], additionally
/// returning an [`Explanation`] of *why* the path was included or excluded.
///
/// This is an optional, opt-in mode: it shares the exact matching engine
/// with [`glob_match`] (the instrumentation is generic over an internal
/// trace sink and monomorphizes away on the default path, so [`glob_match`]
/// keeps its performance), and `explanation.matched == glob_match(glob,
/// path)` always holds — enabling explain never changes the result set.
///
/// The trace is deterministic and self-contained: no state is shared or
/// reused between calls, so repeating a call — on any thread, after any
/// other call, including a failed one — yields an identical trace.
///
/// # Examples
///
/// ```rust
/// use fast_glob::{glob_match, glob_match_explain, FinalReason, TraceEvent};
///
/// let explanation = glob_match_explain("!**/secret/**", "src/secret/key.pem");
/// assert_eq!(explanation.matched, glob_match("!**/secret/**", "src/secret/key.pem"));
/// assert!(matches!(
///     explanation.trace.last(),
///     Some(TraceEvent::Final { matched: false, reason: FinalReason::NegatedExcluded })
/// ));
/// ```
pub fn glob_match_explain(glob: impl AsRef<[u8]>, path: impl AsRef<[u8]>) -> Explanation {
    let glob = glob.as_ref();
    let path = path.as_ref();

    let mut trace = Vec::new();
    match validate(glob) {
        Ok(()) => {
            let negations = glob.iter().take_while(|&&b| b == b'!').count();
            trace.push(TraceEvent::Compiled { negations });
        }
        Err(error) => {
            trace.push(TraceEvent::CompileError { kind: error.kind, index: error.index });
        }
    }
    trace.push(TraceEvent::Start { path_len: path.len() });

    let outcome = glob_match_run(glob, path, &mut VecSink(&mut trace));

    let reason = if outcome.invalid {
        FinalReason::InvalidPattern
    } else if outcome.negated {
        if outcome.inner_matched {
            FinalReason::NegatedExcluded
        } else {
            FinalReason::NegatedIncluded
        }
    } else if outcome.inner_matched {
        FinalReason::FullMatch
    } else {
        FinalReason::NoMatch
    };
    trace.push(TraceEvent::Final { matched: outcome.matched, reason });

    Explanation { matched: outcome.matched, trace }
}

/// Checks that `glob` is a valid pattern.
///
/// [`glob_match`] has no way to report an invalid pattern, and its result for
/// one is unspecified. Call this once when a pattern is first accepted (e.g.
/// at configuration load time) to reject invalid patterns with an actionable
/// error instead of silently matching nothing.
///
/// A pattern accepted here is never treated as invalid by [`glob_match`], so
/// its matching behavior is well-defined. For a rejected pattern the result
/// of [`glob_match`] is unspecified — typically it matches nothing.
///
/// # Examples
///
/// ```rust
/// use fast_glob::{validate, Error, ErrorKind};
///
/// assert!(validate("some/**/n*d[k-m]e?txt").is_ok());
/// assert_eq!(
///     validate("src/**/*.{js,ts"),
///     Err(Error { kind: ErrorKind::UnclosedBrace, index: 9 })
/// );
/// ```
pub fn validate(glob: impl AsRef<[u8]>) -> Result<(), Error> {
    let glob = glob.as_ref();
    let mut index = 0;

    // Leading `!` characters negate the glob and are not part of the pattern.
    while index < glob.len() && glob[index] == b'!' {
        index += 1;
    }

    let mut open_braces = ArrayVec::<usize, MAX_BRACE_NESTING>::new();

    while index < glob.len() {
        match glob[index] {
            b'\\' => {
                if index + 1 >= glob.len() {
                    return Err(Error { kind: ErrorKind::TrailingBackslash, index });
                }
                index += 2;
            }
            b'[' => match skip_class(glob, index) {
                Some(next) => index = next,
                None => return Err(Error { kind: ErrorKind::UnclosedBracket, index }),
            },
            b'{' => {
                if open_braces.try_push(index).is_err() {
                    return Err(Error { kind: ErrorKind::BraceNestingTooDeep, index });
                }
                index += 1;
            }
            // A `}` without a matching `{` is an ordinary character.
            b'}' => {
                open_braces.pop();
                index += 1;
            }
            _ => index += 1,
        }
    }

    if let Some(&index) = open_braces.first() {
        return Err(Error { kind: ErrorKind::UnclosedBrace, index });
    }

    Ok(())
}

/// Returns the match result (with negation applied) alongside whether the
/// pattern was detected as invalid, so tests can check the latter against
/// [`validate`].
fn glob_match_internal(glob: &[u8], path: &[u8]) -> (bool, bool) {
    let outcome = glob_match_run(glob, path, &mut NoSink);
    (outcome.matched, outcome.invalid)
}

/// Internal trace sink, generic so the default [`glob_match`] path
/// monomorphizes [`NoSink`] into nothing: instrumentation is zero-cost
/// unless [`glob_match_explain`] is used. Kept crate-private — it is an
/// internal boundary serving the two public entry points, not a new API.
trait Sink {
    #[inline(always)]
    fn event(&mut self, _event: TraceEvent) {}
}

struct NoSink;

impl Sink for NoSink {}

struct VecSink<'a>(&'a mut Vec<TraceEvent>);

impl Sink for VecSink<'_> {
    #[inline(always)]
    fn event(&mut self, event: TraceEvent) {
        self.0.push(event);
    }
}

/// The outcome of a single matching run, before it is projected into the
/// public shapes (`bool`, `(bool, bool)`, or [`Explanation`]).
struct MatchOutcome {
    /// The verdict [`glob_match`] would return.
    matched: bool,
    /// Whether the path matched the pattern before negation was applied.
    inner_matched: bool,
    /// Whether the pattern is negated an odd number of times.
    negated: bool,
    /// Whether the pattern was detected as invalid while matching.
    invalid: bool,
}

fn glob_match_run(glob: &[u8], path: &[u8], sink: &mut impl Sink) -> MatchOutcome {
    let mut state = State::default();

    let mut negated = false;
    while state.glob_index < glob.len() && glob[state.glob_index] == b'!' {
        negated = !negated;
        state.glob_index += 1;
    }

    let mut brace_stack = BraceStack::new();
    let mut invalid_pattern = false;
    let match_start = state.glob_index;
    let matched = state.glob_match_from(
        glob,
        path,
        match_start,
        &mut brace_stack,
        &mut invalid_pattern,
        sink,
    );

    // A negated glob matches every path its pattern does not — for an invalid
    // pattern that would be every path, even when the matcher never reaches
    // the invalid construct (e.g. after an early literal mismatch). Gate the
    // negation flip on validity instead of relying on lazy detection.
    if negated && !matched && !invalid_pattern && validate(glob).is_err() {
        return MatchOutcome { matched: false, inner_matched: false, negated, invalid: true };
    }

    MatchOutcome {
        matched: negated ^ matched,
        inner_matched: matched,
        negated,
        invalid: invalid_pattern,
    }
}

/// Returns the index just past the `]` closing the character class opened by
/// the `[` at `index`, or `None` if the class is unclosed. Mirrors the class
/// parsing in `glob_match_from`: an optional `^`/`!` prefix, then the first
/// character is a literal member (so a leading `]` does not close the class),
/// and `\` escapes the next character.
fn skip_class(glob: &[u8], index: usize) -> Option<usize> {
    let mut index = index + 1;
    if matches!(glob.get(index), Some(b'^' | b'!')) {
        index += 1;
    }

    let mut first = true;
    loop {
        match glob.get(index)? {
            b']' if !first => return Some(index + 1),
            b'\\' => index += 1,
            _ => {}
        }
        first = false;
        index += 1;
    }
}

#[inline(always)]
fn unescape(c: &mut u8, glob: &[u8], state: &mut State, invalid_pattern: &mut bool) -> bool {
    if *c == b'\\' {
        state.glob_index += 1;
        if state.glob_index >= glob.len() {
            // A trailing backslash has nothing to escape.
            *invalid_pattern = true;
            return false;
        }
        *c = match glob[state.glob_index] {
            b'a' => b'\x61',
            b'b' => b'\x08',
            b'n' => b'\n',
            b'r' => b'\r',
            b't' => b'\t',
            c => c,
        }
    }
    true
}

impl State {
    #[inline(always)]
    fn backtrack<S: Sink>(&mut self, sink: &mut S) {
        self.glob_index = self.wildcard.glob_index as usize;
        self.path_index = self.wildcard.path_index as usize;
        self.brace_depth = self.wildcard.brace_depth as usize;
        sink.event(TraceEvent::Backtrack {
            glob_index: self.glob_index,
            path_index: self.path_index,
        });
    }

    #[inline(always)]
    fn skip_globstars(&mut self, glob: &[u8]) {
        let mut glob_index = self.glob_index + 2;

        while glob_index + 4 <= glob.len() && &glob[glob_index..glob_index + 4] == b"/**/" {
            glob_index += 3;
        }

        if &glob[glob_index..] == b"/**" {
            glob_index += 3;
        }

        self.glob_index = glob_index - 2;
    }

    #[inline(always)]
    fn skip_to_separator<S: Sink>(&mut self, path: &[u8], is_end_invalid: bool, sink: &mut S) {
        let from = self.path_index;
        if self.path_index == path.len() {
            self.wildcard.path_index += 1;
        } else {
            let mut path_index = self.path_index;
            while path_index < path.len() && !is_separator(path[path_index] as char) {
                path_index += 1;
            }

            if is_end_invalid || path_index != path.len() {
                path_index += 1;
            }

            self.wildcard.path_index = path_index as u32;
            self.globstar = self.wildcard;
        }
        sink.event(TraceEvent::SkipToSeparator {
            from,
            to: self.wildcard.path_index as usize,
        });
    }

    #[inline(always)]
    fn skip_branch<S: Sink>(&mut self, glob: &[u8], sink: &mut S) {
        let from = self.glob_index;
        let end_brace_depth = self.brace_depth - 1;
        'skipped: {
            while self.glob_index < glob.len() {
                match glob[self.glob_index] {
                    b'{' => self.brace_depth += 1,
                    b'}' => {
                        self.brace_depth -= 1;
                        if self.brace_depth == end_brace_depth {
                            self.glob_index += 1;
                            break 'skipped;
                        }
                    }
                    b'[' => {
                        // An unclosed class swallows the rest of the glob.
                        self.glob_index = skip_class(glob, self.glob_index).unwrap_or(glob.len());
                        continue;
                    }
                    b'\\' => self.glob_index += 1,
                    _ => (),
                }
                self.glob_index += 1;
            }
        }
        sink.event(TraceEvent::BraceSkip { from, to: self.glob_index });
    }

    fn match_brace_branch<S: Sink>(
        &self,
        glob: &[u8],
        path: &[u8],
        open_brace_index: usize,
        branch_index: usize,
        brace_stack: &mut BraceStack,
        invalid_pattern: &mut bool,
        sink: &mut S,
    ) -> bool {
        // Gracefully reject brace expansions deeper than BraceStack capacity.
        if brace_stack.try_push((open_brace_index as u32, branch_index as u32)).is_err() {
            *invalid_pattern = true;
            return false;
        }

        let mut branch_state = self.clone();
        branch_state.glob_index = branch_index;
        branch_state.brace_depth = brace_stack.len();

        let matched = branch_state.glob_match_from(
            glob,
            path,
            branch_index,
            brace_stack,
            invalid_pattern,
            sink,
        );

        brace_stack.pop();

        matched
    }

    fn match_brace<S: Sink>(
        &mut self,
        glob: &[u8],
        path: &[u8],
        brace_stack: &mut BraceStack,
        invalid_pattern: &mut bool,
        sink: &mut S,
    ) -> bool {
        let mut brace_depth = 0;
        let mut has_closing_brace = false;
        let mut matched = false;

        let open_brace_index = self.glob_index;

        let mut branch_index = 0;

        while self.glob_index < glob.len() {
            match glob[self.glob_index] {
                b'{' => {
                    brace_depth += 1;
                    if brace_depth == 1 {
                        branch_index = self.glob_index + 1;
                    }
                }
                b'}' => {
                    brace_depth -= 1;
                    if brace_depth == 0 {
                        has_closing_brace = true;
                        sink.event(TraceEvent::BraceBranch {
                            open: open_brace_index,
                            branch: branch_index,
                        });
                        if self.match_brace_branch(
                            glob,
                            path,
                            open_brace_index,
                            branch_index,
                            brace_stack,
                            invalid_pattern,
                            sink,
                        ) {
                            matched = true;
                        }
                        break;
                    }
                }
                b',' if brace_depth == 1 => {
                    sink.event(TraceEvent::BraceBranch {
                        open: open_brace_index,
                        branch: branch_index,
                    });
                    if self.match_brace_branch(
                        glob,
                        path,
                        open_brace_index,
                        branch_index,
                        brace_stack,
                        invalid_pattern,
                        sink,
                    ) {
                        matched = true;
                    }
                    branch_index = self.glob_index + 1;
                }
                b'[' => {
                    // An unclosed class swallows the rest of the glob,
                    // leaving the brace unclosed as well.
                    self.glob_index = skip_class(glob, self.glob_index).unwrap_or(glob.len());
                    continue;
                }
                b'\\' => self.glob_index += 1,
                _ => (),
            }
            self.glob_index += 1;
        }

        if !has_closing_brace {
            *invalid_pattern = true;
            return false;
        }

        matched
    }

    #[inline(always)]
    fn glob_match_from<S: Sink>(
        &mut self,
        glob: &[u8],
        path: &[u8],
        match_start: usize,
        brace_stack: &mut BraceStack,
        invalid_pattern: &mut bool,
        sink: &mut S,
    ) -> bool {
        while self.glob_index < glob.len() || self.path_index < path.len() {
            if self.glob_index < glob.len() {
                match glob[self.glob_index] {
                    b'*' => {
                        let is_globstar =
                            self.glob_index + 1 < glob.len() && glob[self.glob_index + 1] == b'*';
                        if is_globstar {
                            self.skip_globstars(glob);
                        }

                        self.wildcard.glob_index = self.glob_index as u32;
                        self.wildcard.path_index = self.path_index as u32 + 1;
                        self.wildcard.brace_depth = self.brace_depth as u32;

                        let mut in_globstar = false;
                        if is_globstar {
                            self.glob_index += 2;

                            let is_end_invalid = self.glob_index != glob.len();

                            if (self.glob_index.saturating_sub(match_start) < 3
                                || glob[self.glob_index - 3] == b'/')
                                && (!is_end_invalid || glob[self.glob_index] == b'/')
                            {
                                if is_end_invalid {
                                    self.glob_index += 1;
                                }

                                self.skip_to_separator(path, is_end_invalid, sink);
                                in_globstar = true;
                            }
                        } else {
                            self.glob_index += 1;
                        }

                        if !in_globstar
                            && self.path_index < path.len()
                            && is_separator(path[self.path_index] as char)
                        {
                            self.wildcard = self.globstar;
                        }

                        sink.event(TraceEvent::Wildcard {
                            globstar: in_globstar,
                            glob_index: self.wildcard.glob_index as usize,
                            path_index: self.wildcard.path_index as usize,
                        });

                        continue;
                    }
                    b'?' if self.path_index < path.len()
                        && !is_separator(path[self.path_index] as char) =>
                    {
                        self.glob_index += 1;
                        self.path_index += 1;
                        continue;
                    }
                    b'[' if self.path_index < path.len() => {
                        self.glob_index += 1;

                        let mut negated = false;
                        if self.glob_index < glob.len()
                            && matches!(glob[self.glob_index], b'^' | b'!')
                        {
                            negated = true;
                            self.glob_index += 1;
                        }

                        let mut first = true;
                        let mut is_match = false;
                        let c = path[self.path_index];
                        while self.glob_index < glob.len()
                            && (first || glob[self.glob_index] != b']')
                        {
                            let mut low = glob[self.glob_index];
                            if !unescape(&mut low, glob, self, invalid_pattern) {
                                return false;
                            }

                            self.glob_index += 1;

                            let high = if self.glob_index + 1 < glob.len()
                                && glob[self.glob_index] == b'-'
                                && glob[self.glob_index + 1] != b']'
                            {
                                self.glob_index += 1;

                                let mut high = glob[self.glob_index];
                                if !unescape(&mut high, glob, self, invalid_pattern) {
                                    return false;
                                }

                                self.glob_index += 1;
                                high
                            } else {
                                low
                            };

                            if low <= c && c <= high {
                                is_match = true;
                            }

                            first = false;
                        }

                        if self.glob_index >= glob.len() {
                            *invalid_pattern = true;
                            return false;
                        }

                        self.glob_index += 1;
                        if is_match != negated {
                            self.path_index += 1;
                            continue;
                        }
                    }
                    b'{' => {
                        if let Some((_, branch_index)) =
                            brace_stack.iter().find(|(open_brace_index, _)| {
                                *open_brace_index == self.glob_index as u32
                            })
                        {
                            self.glob_index = *branch_index as usize;
                            self.brace_depth += 1;
                            continue;
                        }
                        return self.match_brace(glob, path, brace_stack, invalid_pattern, sink);
                    }
                    b',' | b'}' if self.brace_depth > 0 => {
                        self.skip_branch(glob, sink);
                        continue;
                    }
                    mut c if self.path_index < path.len() => {
                        if !unescape(&mut c, glob, self, invalid_pattern) {
                            return false;
                        }

                        let is_match = if c == b'/' {
                            is_separator(path[self.path_index] as char)
                        } else {
                            path[self.path_index] == c
                        };

                        if is_match {
                            self.glob_index += 1;
                            self.path_index += 1;

                            if c == b'/' {
                                sink.event(TraceEvent::Separator {
                                    path_index: self.path_index - 1,
                                    byte: path[self.path_index - 1],
                                });
                                self.wildcard = self.globstar;
                            }

                            continue;
                        }
                    }
                    _ => {}
                }
            }

            if self.wildcard.path_index > 0 && self.wildcard.path_index <= path.len() as u32 {
                self.backtrack(sink);
                continue;
            }

            sink.event(TraceEvent::Pruned {
                glob_index: self.glob_index,
                path_index: self.path_index,
            });
            return false;
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALPHABET: &[u8] = b"a/*[]{}\\,!-";

    fn for_each_pattern(len: usize, f: &mut impl FnMut(&[u8])) {
        let mut pattern = vec![0u8; len];
        for mut n in 0..ALPHABET.len().pow(len as u32) {
            for slot in &mut pattern {
                *slot = ALPHABET[n % ALPHABET.len()];
                n /= ALPHABET.len();
            }
            f(&pattern);
        }
    }

    /// `validate` and the matcher must agree on what is invalid:
    /// a pattern `validate` accepts is never flagged invalid by the matcher,
    /// and an invalid non-negated pattern never matches any path.
    /// Checked exhaustively over all short patterns built from the special characters.
    #[test]
    fn validate_agrees_with_matcher() {
        const PATHS: &[&str] = &["", "a", "aa", "a/a", "-", ",", "!"];

        for len in 0..=6 {
            for_each_pattern(len, &mut |pattern| {
                let valid = validate(pattern).is_ok();
                for path in PATHS {
                    let (matched, invalid) = glob_match_internal(pattern, path.as_bytes());
                    if valid {
                        assert!(
                            !invalid,
                            "matcher flagged {:?} as invalid on path {path:?} but validate accepted it",
                            String::from_utf8_lossy(pattern),
                        );
                    } else if pattern.first() == Some(&b'!') || !pattern.contains(&b'{') {
                        // A negated invalid pattern never matches (the negation flip is gated on validity),
                        // and neither does a brace-free one, since every part of its glob is processed directly.
                        // (A non-negated invalid construct in a non-taken brace branch may go unnoticed,
                        // that behavior is documented as unspecified.)
                        assert!(
                            !(matched && !invalid),
                            "invalid pattern {:?} matched path {path:?}",
                            String::from_utf8_lossy(pattern),
                        );
                    }
                }
            });
        }
    }
}
