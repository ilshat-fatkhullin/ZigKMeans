const std = @import("std");
const Allocator = std.mem.Allocator;

pub const KMeans = struct {
    allocator: Allocator,
    k: usize,
    centroids: [][]f32,
    seed: u64,

    pub fn init(k: usize, seed: ?u64) KMeans {
        return KMeans{
            .allocator = std.heap.page_allocator,
            .k = k,
            .centroids = &[_][]f32{},
            .seed = seed orelse 0,
        };
    }

    pub fn deinit(self: *KMeans) void {
        self.deinit_centroids();
    }

    pub fn fit(self: *KMeans, data: [][]f32) void {
        const allocator = self.allocator;

        const k = self.k;

        const rows = data.len;
        const columns = data[0].len;

        self.deinit_centroids();
        self.centroids = allocator.alloc([]f32, k) catch unreachable;
        var new_centroids: [][]f32 = allocator.alloc([]f32, k) catch unreachable;
        const counts: []usize = allocator.alloc(usize, k) catch unreachable;

        defer allocator.free(new_centroids);
        defer allocator.free(counts);

        for (0..k) |i| {
            self.centroids[i] = allocator.alloc(f32, columns) catch unreachable;
            new_centroids[i] = allocator.alloc(f32, columns) catch unreachable;
        }

        // Initialization: choose k centroids randomly from the dataset
        var prng = std.Random.DefaultPrng.init(self.seed);
        for (0..k) |i| {
            const index = prng.random().uintLessThan(usize, rows);
            std.mem.copyForwards(f32, self.centroids[i], data[index]);
        }

        // Loop until convergence
        var converged: bool = false;

        while (!converged) {
            // Step 1: Assign each point to the nearest centroid
            var labels: []usize = allocator.alloc(usize, rows) catch unreachable;
            defer allocator.free(labels);

            for (0..rows) |i| {
                labels[i] = self.closest_centroid(data[i]);
            }

            // Step 2: Reset new centroids and counts
            for (0..k) |i| {
                for (0..columns) |j| {
                    new_centroids[i][j] = 0;
                }
                counts[i] = 0;
            }

            // Step 3: Update centroids
            for (0..rows) |i| {
                const label = labels[i];
                add(new_centroids[label], data[i]);
                counts[label] += 1;
            }

            for (0..(self.k)) |i| {
                if (counts[i] > 0) {
                    divide(new_centroids[i], counts[i]);
                }
            }

            // Check for convergence
            converged = true;
            for (0..(self.k)) |i| {
                if (!std.mem.eql(f32, self.centroids[i], new_centroids[i])) {
                    converged = false;
                    break;
                }
            }

            // Update centroids
            for (0..k) |i| {
                std.mem.copyForwards(f32, self.centroids[i], new_centroids[i]);
            }
        }

        // Free the new centroids
        for (new_centroids) |centroid| {
            allocator.free(centroid);
        }
    }

    pub fn predict_single(self: KMeans, point: []f32) usize {
        return self.closest_centroid(point);
    }

    pub fn predict_multiple(self: KMeans, points: [][]f32) []usize {
        const labels: []usize = self.allocator.alloc(usize, points.len) catch unreachable;
        for (0..points.len) |i| {
            labels[i] = self.closest_centroid(points[i]);
        }
        return labels;
    }

    fn deinit_centroids(self: *KMeans) void {
        const allocator = self.allocator;
        const centroids = self.centroids;

        if (self.centroids.len > 0) {
            for (self.centroids) |centroid| {
                if (centroid.len > 0) {
                    allocator.free(centroid);
                }
            }
            allocator.free(centroids);
            self.centroids = &[_][]f32{};
        }
    }

    fn closest_centroid(self: KMeans, point: []f32) usize {
        var min_distance: f32 = std.math.floatMax(f32);
        var closest: usize = 0;

        for (0..self.k) |i| {
            const distance = square_distance(point, self.centroids[i]);
            if (distance < min_distance) {
                min_distance = distance;
                closest = i;
            }
        }

        return closest;
    }

    fn square_distance(a: []f32, b: []f32) f32 {
        var sum: f32 = 0;
        for (0..a.len) |i| {
            const diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    fn add(a: []f32, b: []f32) void {
        for (0..a.len) |i| {
            a[i] += b[i];
        }
    }

    fn divide(a: []f32, b: usize) void {
        for (0..a.len) |i| {
            a[i] /= @floatFromInt(b);
        }
    }
};

test "fit and predict" {
    const raw_data = [_][2]f32{
        // Cluster 1
        .{ -4.5, -5 },
        .{ -5, -5.5 },
        .{ -5, -5 },
        .{ -5, -6 },
        // Cluster 2
        .{ 4.5, 5 },
        .{ 5, 5.5 },
        .{ 5, 5 },
        .{ 5, 6 },
    };

    const allocator = std.heap.page_allocator;

    const dataset: [][]f32 = allocator.alloc([]f32, raw_data.len) catch unreachable;
    defer allocator.free(dataset);

    var index: usize = 0;
    for (raw_data) |point| {
        dataset[index] = allocator.alloc(f32, point.len) catch unreachable;
        for (0..point.len) |j| {
            dataset[index][j] = point[j];
        }
        index += 1;
    }

    var k_means = KMeans.init(2, 0);
    defer k_means.deinit();

    k_means.fit(dataset);

    // Assert centroids are close to expected values
    const expected_centroids = [_][2]f32{
        .{ 4.88, 5.38 },
        .{ -4.88, -5.38 },
    };

    for (0..k_means.k) |i| {
        const centroid = k_means.centroids[i];
        const expected_centroid = expected_centroids[i];

        for (0..centroid.len) |j| {
            try std.testing.expectApproxEqAbs(expected_centroid[j], centroid[j], 0.1);
        }
    }

    // Dealloc dataset
    for (dataset) |point| {
        allocator.free(point);
    }

    // Test prediction
    const raw_points = [_][2]f32{
        .{ 5, 5 },
        .{ -1, -1 },
        .{ 3, 2 },
    };

    const points: [][]f32 = allocator.alloc([]f32, raw_points.len) catch unreachable;
    defer allocator.free(points);

    index = 0;
    for (raw_points) |point| {
        points[index] = allocator.alloc(f32, point.len) catch unreachable;
        for (0..point.len) |j| {
            points[index][j] = point[j];
        }
        index += 1;
    }

    const expected_labels = [_]usize{
        0,
        1,
        0,
    };

    const labels = k_means.predict_multiple(points);
    defer allocator.free(labels);
    for (0..points.len) |i| {
        try std.testing.expect(labels[i] == expected_labels[i]);
    }

    const label = k_means.predict_single(points[0]);
    try std.testing.expect(label == expected_labels[0]);

    // Dealloc points
    for (points) |point| {
        allocator.free(point);
    }
}
