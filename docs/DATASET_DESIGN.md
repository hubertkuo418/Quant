# Dataset Design

## Shapes

- Input: `[batch, sequence_length=60, feature_dim=13]`
- Target: `[batch, horizons=3]`
- Horizons: forward 5-, 20-, and 60-trading-day adjusted-close returns

Each sequence contains one ticker only and ends on the prediction date.

## Split Rules

The default fixed experiment uses:

- Train realization cutoff: `2021-12-31`
- Validation realization cutoff: `2023-12-31`
- Test feature dates: after `2023-12-31`

A training row is accepted only when its 60-day target realization date is on
or before the train cutoff. Validation follows the same rule at its cutoff.
Rows whose labels cross a boundary remain `unassigned`; this purging prevents
future-period returns from leaking into model fitting or selection.

## Normalization

For every configured feature:

1. Fit the missing-value median on rows dated on or before the train cutoff.
2. Impute missing values with that training median.
3. Fit mean and population standard deviation on the imputed training values.
4. Apply those frozen statistics to train, validation, and test.

An entirely missing optional feature receives median `0`, mean `0`, and scale
`1`. This keeps the MVP runnable before a point-in-time fundamentals source is
configured without inventing information.

## Saved Artifacts

- Normalized panel with targets and split labels.
- JSON scaler parameters.
- Dataset manifest with feature names, target names, and sample counts.
- In-memory PyTorch `Dataset` objects and configurable `DataLoader` instances.
