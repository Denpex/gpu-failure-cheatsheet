# Contributing

Signatures and corrections are welcome. The bar is evidence, not volume.

## Adding a failure signature

Open a pull request editing `signatures.json`, or an issue with the same information
if you would rather not edit JSON.

A record looks like this:

```json
{
  "slug": "nccl-ib-hca-mismatch",
  "title": "NCCL selects the wrong InfiniBand HCA",
  "category": "Network",
  "severity": "high",
  "signatures": ["NCCL WARN Could not find real path of mlx5_1"]
}
```

Add `"xid": 79` when the failure is identified by an NVIDIA Xid code.

## The one hard rule

**Every string in `signatures` must be text a machine actually printed, copied from a
log you or a cited source really saw.**

Do not write a line that looks like what the error probably says. A fabricated
signature still matches real logs, so it does not fail loudly. It sends somebody
debugging a production outage to the wrong root cause, which is worse than this table
not covering their failure at all.

If you have the failure but not the exact output, open an issue describing it rather
than guessing at the text. That is a useful contribution too.

## What makes a good signature

- **Specific.** `CUDA out of memory` matches almost every training log ever written.
  `CUDA out of memory. Tried to allocate 2.00 GiB` identifies something.
- **Stable across runs.** Strip ranks, PIDs, timestamps, hostnames and hex addresses.
  Keep numbers that ARE the signal: `Xid 79`, `vendor err 129`, `exit code 137`.
- **One line.** The matcher works line by line and normalises whitespace.

## Checks

CI runs `node triage.mjs sample.log`, validates `signatures.json`, and checks the
`--json` output parses. Run those locally before opening the PR:

```bash
node triage.mjs sample.log
node triage.mjs sample.log --json
```

## Licence

Code is MIT. Data and documentation are CC BY 4.0. Contributions are accepted under
those same terms.
