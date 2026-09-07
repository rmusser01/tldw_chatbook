# Actor Pack import golden fixtures

Import compatibility is intentionally verified against the committed archives in
`../export-golden/`. Those archives are treated as independent, immutable inputs:
the import tests open them directly and do not regenerate them through the import
implementation.

The shared fixtures cover minimal Character and Persona packs. Section-bearing
and adversarial archives are assembled per test so each manifest, asset, and
failure mutation remains visible beside its assertion.
