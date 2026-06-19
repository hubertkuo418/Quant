# Investor-Needs Profiles

The Strategy Studio questionnaire converts user preferences into explicit,
reviewable recommendation constraints. It does not claim to provide regulated
personal investment advice.

## Inputs

- conservative, balanced, or aggressive risk tolerance;
- maximum acceptable drawdown;
- low, medium, or flexible turnover preference;
- minimum historical annual return;
- short, medium, or long intended holding period;
- strict, balanced, or flexible execution robustness;
- optional mandatory OOS evidence;
- number of candidates to return.

## Evidence Behavior

Profiles store minimum `oos_sharpe` and `robustness_pass_rate` targets. When
strict evidence mode is enabled, recommendation candidates missing either field
are rejected. When it is disabled, the legacy common-period optimizer can still
be ranked, but the result must not be described as OOS-qualified.

The next platform step is to generate Walk-forward and robustness evidence for
each Pareto candidate, allowing strict evidence mode to become the default.
