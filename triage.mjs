#!/usr/bin/env node
/**
 * Match a crash log against known GPU failure signatures.
 *
 * Zero dependencies, no network, reads nothing but the file you name. Run it on an
 * air-gapped cluster if you like; it is a substring matcher over a bundled table, not
 * a client for anything.
 *
 *   node triage.mjs /path/to/training.log
 *   cat training.log | node triage.mjs
 *
 * Exit codes: 0 a match was found, 1 no match, 2 usage or read error.
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const SITE = 'https://denpex.com';

// --json makes this embeddable in other people's tooling, which is how a reference
// spreads without anyone promoting it. Parsed before the file argument so that
// `triage.mjs --json run.log` and `triage.mjs run.log --json` both work; a flag that
// only works in one position is a flag people give up on.
const ARGS = process.argv.slice(2);
const JSON_OUT = ARGS.includes('--json');
const FILE_ARG = ARGS.find((a) => !a.startsWith('--'));

function readInput() {
  if (FILE_ARG && FILE_ARG !== '-') return readFileSync(FILE_ARG, 'utf8');
  if (process.stdin.isTTY) {
    console.error('usage: node triage.mjs <logfile> [--json]   (or pipe a log on stdin)');
    process.exit(2);
  }
  return readFileSync(0, 'utf8');
}

function loadSignatures() {
  try {
    return JSON.parse(readFileSync(join(HERE, 'signatures.json'), 'utf8'));
  } catch {
    console.error('Could not read signatures.json. Download it next to triage.mjs.');
    process.exit(2);
  }
}

/**
 * Normalise so run-specific noise does not defeat a substring match.
 *
 * DIGITS ARE KEPT ON PURPOSE. An earlier version replaced every integer with a space,
 * which reads as sensible noise-stripping and is precisely wrong here: "Xid 79",
 * "vendor err 129" and "exit code 137" are signatures whose whole content is the
 * number. That version matched eight NCCL timeouts in a test log and missed the Xid 79
 * that had caused all of them, which is the exact misreading this repo exists to warn
 * about. Only genuinely run-specific tokens are stripped: timestamps, hex, PCI
 * addresses and rank or pid identifiers.
 */
function normalise(text) {
  return text
    .toLowerCase()
    .replace(/\d{4}-\d{2}-\d{2}[t ]\d{2}:\d{2}:\d{2}[.,\d]*z?/g, ' ')
    .replace(/\(pci:[^)]*\)/g, ' ')
    .replace(/\b[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]\b/g, ' ')
    .replace(/0x[0-9a-f]+/g, ' ')
    .replace(/\b(rank|pid|gpu|node)[-\s:=]*\d+\b/g, ' ')
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

/** Every Xid code the log reports. Addresses are stripped first so the digits nearest
 *  "Xid" are the code rather than a bus address. */
function xidCodesIn(text) {
  const cleaned = text
    .toLowerCase()
    .replace(/\(pci:[^)]*\)/g, ' ')
    .replace(/\b[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]\b/g, ' ')
    .replace(/0x[0-9a-f]+/g, ' ');
  const found = new Set();
  for (const m of cleaned.matchAll(/\bxid\b\D{0,20}?(\d{1,3})\b/g)) {
    found.add(Number(m[1]));
  }
  return found;
}

const log = readInput();
const haystack = normalise(log);
const loggedXids = xidCodesIn(log);
const records = loadSignatures();

const hits = [];
for (const record of records) {
  const matched = (record.signatures ?? []).filter((sig) => {
    const needle = normalise(sig);
    return needle.length >= 8 && haystack.includes(needle);
  });
  const xidHit = typeof record.xid === 'number' && loggedXids.has(record.xid);
  if (xidHit) matched.unshift('Xid ' + record.xid + ' reported in this log');
  if (matched.length > 0) hits.push({ record, matched, xidHit });
}

// An Xid is a statement about hardware and outranks a framework-level symptom, which
// in a distributed run is usually downstream of it. Beyond that, more matched
// signatures is a stronger match.
const SEVERITY = { critical: 3, high: 2, medium: 1, low: 0 };
hits.sort((a, b) => (
  Number(b.xidHit) - Number(a.xidHit)
  || b.matched.length - a.matched.length
  || (SEVERITY[b.record.severity] ?? 0) - (SEVERITY[a.record.severity] ?? 0)
  // Slug is the tiebreaker so a run is reproducible. Without it, two records with
  // identical scores order by however the JSON happened to load.
  || a.record.slug.localeCompare(b.record.slug)
));

/**
 * COLLAPSE RECORDS THAT REPORT THE SAME FINDING.
 *
 * The corpus carries five separate write-ups of Xid 79. They are different pages with
 * different sourcing, which is fine on the site, but at a terminal they all match the
 * same line and score identically, so the top 8 became five restatements of one fact
 * and three real findings. On the bundled sample.log that is exactly what happened.
 *
 * That is worse than noise. A triage tool whose first screen is the same answer five
 * times reads as keyword padding, which is the opposite of what a reference is for, and
 * it buries the NCCL timeouts that tell you how far the fault spread.
 *
 * So group by the FINDING rather than the page. An Xid code is the finding; for
 * everything else the strongest matched signature is. The best-scoring record in each
 * group represents it and the rest are reported as a count, so nothing is hidden.
 */
function findingKey(hit) {
  if (hit.xidHit) return 'xid:' + hit.record.xid;
  return 'sig:' + normalise(hit.matched[0] ?? hit.record.slug);
}

const groups = new Map();
for (const hit of hits) {
  const key = findingKey(hit);
  const group = groups.get(key);
  // `hits` is already sorted, so the first arrival is the best representative.
  if (group) group.alternates += 1;
  else groups.set(key, { ...hit, alternates: 0 });
}
const findings = [...groups.values()];

// Machine-readable output, so this can be a step in someone else's pipeline rather
// than only a thing a human reads. Emitted before the no-match early exit so a clean
// log still produces valid JSON rather than prose on stdout.
if (JSON_OUT) {
  console.log(JSON.stringify({
    matchedSignatures: hits.length,
    distinctFindings: findings.length,
    findings: findings.map((f) => ({
      title: f.record.title,
      slug: f.record.slug,
      category: f.record.category,
      severity: f.record.severity,
      hardware: f.xidHit,
      ...(typeof f.record.xid === 'number' ? { xid: f.record.xid } : {}),
      matched: f.matched,
      alternateWriteUps: f.alternates,
      details: SITE + '/failures/' + f.record.slug,
    })),
  }, null, 2));
  process.exit(hits.length > 0 ? 0 : 1);
}

if (hits.length === 0) {
  console.log('No known signature matched.');
  console.log('');
  console.log('That does not mean nothing is wrong. This table only covers failures with a');
  console.log('published verbatim signature, and it matches on text rather than reasoning about');
  console.log('cross-rank ordering, so it cannot tell you which rank failed first.');
  console.log('');
  console.log('Paste the full log at ' + SITE + '/console for a real diagnosis.');
  process.exit(1);
}

const SHOWN = 8;
const shown = findings.slice(0, SHOWN);

// Report both numbers. "5 findings across 16 matched signatures" is the honest summary:
// it says how much collapsing happened without making you scroll through it.
const summary = findings.length === hits.length
  ? 'Matched ' + hits.length + ' known failure' + (hits.length === 1 ? '' : 's')
  : 'Matched ' + findings.length + ' distinct finding' + (findings.length === 1 ? '' : 's')
    + ' across ' + hits.length + ' known signatures';
console.log(summary + (findings.length > SHOWN ? ', showing the top ' + SHOWN : '') + ':');
console.log('');
for (const hit of shown) {
  console.log('  ' + (hit.xidHit ? '[hardware] ' : '') + hit.record.title);
  console.log('    category  ' + hit.record.category + ', severity ' + hit.record.severity);
  console.log('    matched   ' + JSON.stringify(hit.matched[0]));
  console.log('    details   ' + SITE + '/failures/' + hit.record.slug);
  if (hit.alternates > 0) {
    console.log(
      '    also      ' + hit.alternates + ' other entr' + (hit.alternates === 1 ? 'y describes' : 'ies describe')
      + ' this same failure',
    );
  }
  console.log('');
}

if (findings.length > 1) {
  console.log('More than one matched. In a distributed run that is expected: the initiating');
  console.log('fault and the collective timeouts it caused both appear in the same log. The');
  console.log('earliest anomalous event is usually the cause, not the most frequent one.');
  if (hits.some((h) => h.xidHit)) {
    console.log('');
    console.log('An Xid was reported, so it is listed first. A hardware fault on one rank will');
    console.log('stall the collective and produce a timeout on every other rank, which is why');
    console.log('the timeouts outnumber it.');
  }
  console.log('');
}
console.log('This is a text matcher. For cross-rank causal ordering, paste the full log at');
console.log(SITE + '/console');
