# Usage

```zig
var k_means = KMeans.init(2, null);
const dataset: [][]f32 = (..build dataset..)
k_means.fit(dataset);
const labels: []usize = k_means.predict_multiple(dataset);
const label: usize = k_means.predict_single(dataset[0]);
```
