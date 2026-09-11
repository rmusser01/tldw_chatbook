# Canvas controls

Use this topic for an accepted interactive artifact. Required profile for the
complete example below: `canvas-v1`. Loading this guide does not establish that a
profile is available; current tool/profile guidance and runtime checks take
precedence, especially for historical edits.

Use explicit `document.getElementById` lookups, supported `addEventListener`
events, control `value`, and `textContent` for visible output. Place classic
scripts after their controls. Keep each input/change callback small and bounded;
validate empty, non-finite, and out-of-range values rather than trusting HTML
constraints alone. Use plain labels, keyboard-operable native inputs, and visible
focus. State stays in the allowed JavaScript realm for this load.

## Complete multiplication example

Change quantity or unit price to update the total. The initial 2 units at 12.50
produce 25.00; 3 units at 12.50 produce 37.50. Invalid input shows a correction
message instead of a misleading number. This is a sample estimate, not a stored
transaction.

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Quantity and price</title>
<style>
body { margin: 0; padding: 16px; font-family: sans-serif; color: #182536; background: #ffffff; }
main { max-width: 480px; margin: 0 auto; }
h1 { font-size: 24px; }
p { line-height: 1.5; }
label { display: block; margin: 20px 0 8px; }
input { box-sizing: border-box; width: 100%; padding: 10px; font-size: 18px; border: 1px solid #536579; border-radius: 4px; }
input:focus { outline: 3px solid #245d91; outline-offset: 2px; }
output { font-size: 22px; font-weight: bold; }
</style>
</head>
<body>
<main>
<h1>Estimate a total</h1>
<p>Enter up to 1000 whole units and a unit price from 0 to 10000.</p>
<label for="quantity">Quantity (whole units)</label>
<input id="quantity" type="number" min="0" max="1000" step="1" value="2">
<label for="unit-price">Unit price</label>
<input id="unit-price" type="number" min="0" max="10000" step="0.01" value="12.50">
<p>Total: <output id="total" for="quantity unit-price" aria-live="polite">25.00</output></p>
</main>
<script>
const quantity = document.getElementById("quantity");
const unitPrice = document.getElementById("unit-price");
const total = document.getElementById("total");
function updateTotal() {
  const count = Number(quantity.value);
  const price = Number(unitPrice.value);
  if (quantity.value.trim() === "" || unitPrice.value.trim() === "" ||
      !Number.isFinite(count) || !Number.isFinite(price) ||
      !Number.isInteger(count) || count < 0 || count > 1000 ||
      price < 0 || price > 10000) {
    total.textContent = "Enter values within the stated limits.";
    return;
  }
  total.textContent = (count * price).toFixed(2);
}
quantity.addEventListener("input", updateTotal);
quantity.addEventListener("change", updateTotal);
unitPrice.addEventListener("input", updateTotal);
unitPrice.addEventListener("change", updateTotal);
updateTotal();
</script>
</body>
</html>
```

Do not use inline event attributes, markup sinks such as `innerHTML`, native
`window`, modules, external libraries, networking, storage, CSS variables, or
HTML canvas drawing APIs. This facade is not a full browser DOM. Use supported
classic scripts and passive SVG; do not assume arbitrary Mermaid.js APIs.
`canvas.submit` and `canvas.download` only request confirmed host actions.

For an edit, read the current source and parent revision first, then send a
complete replacement. Compiler acceptance or a staged/saved revision does not
prove the controls executed in a browser. Report preview readiness only from
received evidence. Consult `repair` for a concrete failure; do not loop through
speculative replacements.
