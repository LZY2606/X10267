use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use fast_glob::{
    explain_walk, walk, InvocationId, PathSeparator, PathSource, PermissionPolicy, PruneReason,
    SymlinkDecision, TraceEvent, WalkErrorKind, WalkOptions,
};

struct TempTree {
    path: PathBuf,
}

impl TempTree {
    fn new(prefix: &str) -> Self {
        let unique = format!(
            "{prefix}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    fn dir(&self, relative: &str) -> &Path {
        let path = self.path.join(relative);
        fs::create_dir_all(&path).unwrap();
        path.as_path()
    }

    fn file(&self, relative: &str) -> &Path {
        let path = self.path.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&path, relative).unwrap();
        path.as_path()
    }

    fn symlink(&self, target: &str, relative: &str) {
        #[cfg(unix)]
        {
            if let Some(parent) = self.path.join(relative).parent() {
                fs::create_dir_all(parent).unwrap();
            }
            std::os::unix::fs::symlink(self.path.join(target), self.path.join(relative)).unwrap();
        }
        #[cfg(not(unix))]
        {
            let _ = (target, relative);
        }
    }
}

impl Drop for TempTree {
    fn drop(&mut self) {
        fn reset(path: &Path) {
            #[cfg(unix)]
            {
                let _ = fs::set_permissions(path, std::os::unix::fs::PermissionsExt::from_mode(0o700));
            }
            #[cfg(not(unix))]
            {
                let _ = path;
            }
        }

        fn remove(path: &Path) {
            reset(path);
            if let Ok(entries) = fs::read_dir(path) {
                for entry in entries.flatten() {
                    remove(&entry.path());
                }
            }
            let _ = fs::remove_dir_all(path);
        }

        remove(&self.path);
    }
}

fn relative_paths(report: &fast_glob::ExplainReport) -> Vec<String> {
    report.paths.iter().map(|path| path.relative.clone()).collect()
}

fn first_violation<'a, T>(
    label: &str,
    actual: impl IntoIterator<Item = T>,
    expected: impl IntoIterator<Item = &'a str>,
) -> Option<String>
where
    T: AsRef<str>,
{
    let actual: Vec<&str> = actual.into_iter().map(|value| value.as_ref()).collect();
    let expected: Vec<&str> = expected.into_iter().collect();
    (actual != expected).then(|| format!("{label}: expected {expected:?}, got {actual:?}"))
}

fn assert_report_consistency(label: &str, report: &fast_glob::ExplainReport) {
    let final_events = report
        .trace
        .iter()
        .filter_map(|event| match event {
            TraceEvent::FinalPath(path) => Some(path.relative.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    if let Some(violation) = first_violation(
        &format!("{label}: final trace events must equal result paths"),
        final_events,
        relative_paths(report),
    ) {
        panic!("{violation}\nfull trace: {:#?}", report.trace);
    }

    let mut event_paths = BTreeMap::new();
    for event in &report.trace {
        let relative = match event {
            TraceEvent::DirectoryPruned(event) => &event.relative,
            TraceEvent::IgnoreHit(event) => &event.relative,
            TraceEvent::SymlinkDecision(event) => &event.relative,
            TraceEvent::FinalPath(event) => &event.relative,
            TraceEvent::ExcludedPath(event) => &event.relative,
            TraceEvent::PermissionDenied(event) => &event.relative,
            _ => continue,
        };
        *event_paths.entry(relative.clone()).or_insert(0u32) += 1;
    }
    assert!(event_paths.values().all(|count| *count >= 1), "{label}: trace has paths without events");
}

fn fixed_options() -> WalkOptions {
    WalkOptions::default()
        .include_hidden(true)
        .permission_policy(PermissionPolicy::Continue)
}

fn fixed_patterns() -> Vec<&'static str> {
    vec![
        "**/*.txt",
        "**/KEEP.TXT",
        "!ignored/**",
        "!**/excluded.txt",
        "nested/**",
        "!ignored/**",
    ]
}

#[test]
fn explain_switch_preserves_result_set_for_fixed_fixture() {
    let tree = TempTree::new("fast-glob-explain-fixed");
    tree.file("nested/keep.txt");
    tree.file("nested/excluded.txt");
    tree.file(".hidden/secret.txt");
    tree.file("ignored/a.txt");
    tree.file("ignored/nested/b.txt");
    tree.file("literal/keep.txt");
    tree.file("literal/skip.log");
    tree.file(r"back\slash.txt");
    tree.dir("cycle");
    tree.symlink("cycle", "cycle/back");

    let no_access = tree.path.join("no-access");
    fs::create_dir_all(&no_access).unwrap();
    fs::write(no_access.join("denied.txt"), "denied").unwrap();
    #[cfg(unix)]
    fs::set_permissions(&no_access, std::os::unix::fs::PermissionsExt::from_mode(0o000)).unwrap();

    let patterns = fixed_patterns();
    let mut paths = walk(&tree.path, patterns.clone(), fixed_options()).unwrap();
    paths.sort();
    let report = explain_walk(&tree.path, patterns, fixed_options()).unwrap();
    let mut explained = report
        .paths
        .iter()
        .map(|path| path.path.clone())
        .collect::<Vec<_>>();
    explained.sort();
    assert_eq!(paths, explained, "explain mode changed the traversal result set");

    let mut relatives = relative_paths(&report);
    relatives.sort();
    let expected = vec![
        ".hidden/secret.txt",
        "back\\slash.txt",
        "literal/keep.txt",
        "nested/keep.txt",
    ];
    let mut expected = expected;
    expected.sort();
    if let Some(violation) = first_violation("fixed fixture result set", relatives, expected.iter().map(String::as_str)) {
        panic!("{violation}\nroot: {}\ntrace: {:#?}", tree.path.display(), report.trace);
    }

    assert_report_consistency("fixed fixture", &report);

    assert!(
        report.trace.iter().any(|event| matches!(
            event,
            TraceEvent::PatternCompiled(compiled)
                if compiled.index == 0 && compiled.effective == "**/*.txt"
        )),
        "pattern compilation trace missing: {:#?}",
        report.trace
    );
    assert!(
        report.trace.iter().any(|event| matches!(
            event,
            TraceEvent::StartDirectory(start) if start.root == tree.path
        )),
        "start directory trace missing: {:#?}",
        report.trace
    );
    assert!(
        report.trace.iter().any(|event| matches!(
            event,
            TraceEvent::DirectoryPruned(pruned)
                if pruned.relative == "ignored" && pruned.reason == PruneReason::NoPatternCanMatch
        )),
        "directory pruning trace missing: {:#?}",
        report.trace
    );
    assert!(
        report.trace.iter().any(|event| matches!(
            event,
            TraceEvent::IgnoreHit(hit) if hit.relative == "ignored" && hit.pattern_index == 2
        )),
        "ignore hit trace missing: {:#?}",
        report.trace
    );
    assert!(
        report.trace.iter().any(|event| matches!(
            event,
            TraceEvent::IgnoreHit(hit) if hit.relative == "nested/excluded.txt"
        )),
        "file ignore hit trace missing: {:#?}",
        report.trace
    );
    assert!(
        report.trace.iter().any(|event| matches!(
            event,
            TraceEvent::SymlinkDecision(symlink)
                if symlink.relative == "cycle/back"
                    && symlink.decision == SymlinkDecision::NotFollowed
        )),
        "symlink decision trace missing: {:#?}",
        report.trace
    );
    assert!(
        report.trace.iter().any(|event| matches!(
            event,
            TraceEvent::FinalPath(final_path)
                if final_path.relative == "nested/keep.txt"
                    && final_path.source == PathSource::File
                    && final_path.matched_pattern_index == 0
        )),
        "final path source trace missing: {:#?}",
        report.trace
    );
}
