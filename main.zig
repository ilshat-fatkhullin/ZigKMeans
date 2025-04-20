const std = @import("std");
const KMeans = @import("k_means.zig").KMeans;

pub fn main() !void {
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

    var k_means = KMeans.init(2, null);
    defer k_means.deinit();

    k_means.fit(dataset);

    //print the centroids
    for (0..k_means.k) |i| {
        std.debug.print("Centroid {}:\n", .{i});

        for (0..k_means.centroids[i].len) |j| {
            std.debug.print("  {}: {d:.2}\n", .{ j, k_means.centroids[i][j] });
        }
    }

    // Deallocate the dataset
    for (0..dataset.len) |i| {
        allocator.free(dataset[i]);
    }
}
