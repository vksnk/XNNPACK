// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Tests for make_split_factors: distribution of the tile area between the
// dimensions, the effect of the loop order, and alignment reservations.

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/subgraph/slinky.h"

namespace ynn {
namespace {

// All tests use element_cost = 4 (f32), i.e. a tile area of 32768 elements.
constexpr int64_t kElementCost = 4;
constexpr int64_t kTileArea = 32768;

int64_t constant(const slinky::expr& e) {
  auto c = slinky::as_constant(e);
  return c ? *c : -1;
}

TEST(MakeSplitFactors, DefaultOrderIsInnermostFirst) {
  slinky_globals globals;
  std::vector<slinky::expr> extents = {16384, 256};
  std::vector<slinky::expr> splits =
      make_split_factors(globals, extents, kElementCost);
  EXPECT_EQ(constant(splits[0]), 16384);
  EXPECT_EQ(constant(splits[1]), kTileArea / 16384);
}

TEST(MakeSplitFactors, LoopOrderGetsFirstClaim) {
  slinky_globals globals;
  std::vector<slinky::expr> extents = {16384, 256};
  std::vector<int> order = {1, 0};
  std::vector<slinky::expr> alignments = {16};
  std::vector<slinky::expr> splits = make_split_factors(
      globals, extents, kElementCost, /*given_splits=*/{}, order, alignments);
  // Dimension 1 is visited first and fits in a tile whole; dimension 0 gets
  // the remaining area as whole alignment-sized blocks.
  EXPECT_EQ(constant(splits[1]), 256);
  EXPECT_EQ(constant(splits[0]), kTileArea / 256);
}

TEST(MakeSplitFactors, AlignmentReservesMinimalSplit) {
  slinky_globals globals;
  std::vector<slinky::expr> extents = {256, 1 << 20};
  std::vector<int> order = {1, 0};
  std::vector<slinky::expr> alignments = {16};
  std::vector<slinky::expr> splits = make_split_factors(
      globals, extents, kElementCost, /*given_splits=*/{}, order, alignments);
  // Dimension 1 is visited first, but can't consume the whole area: one
  // alignment-sized block stays reserved for dimension 0.
  EXPECT_EQ(constant(splits[1]), kTileArea / 16);
  EXPECT_EQ(constant(splits[0]), 16);
}

TEST(MakeSplitFactors, SplitIsMultipleOfAlignment) {
  slinky_globals globals;
  std::vector<slinky::expr> extents = {1 << 20, 3};
  std::vector<int> order = {1, 0};
  std::vector<slinky::expr> alignments = {16};
  std::vector<slinky::expr> splits = make_split_factors(
      globals, extents, kElementCost, /*given_splits=*/{}, order, alignments);
  EXPECT_EQ(constant(splits[1]), 3);
  // Dimension 0 gets the area left over from the odd-sized dimension 1,
  // rounded down to a multiple of the alignment.
  EXPECT_GT(constant(splits[0]), 0);
  EXPECT_EQ(constant(splits[0]) % 16, 0);
  EXPECT_EQ(constant(splits[0]), kTileArea / (16 * 3) * 16);
}

TEST(MakeSplitFactors, ReservationIsCappedByExtent) {
  slinky_globals globals;
  std::vector<slinky::expr> extents = {4, 65536};
  std::vector<int> order = {1, 0};
  std::vector<slinky::expr> alignments = {16};
  std::vector<slinky::expr> splits = make_split_factors(
      globals, extents, kElementCost, /*given_splits=*/{}, order, alignments);
  // Dimension 0 can only ever use 4 elements of the area, so its reservation
  // must not shrink dimension 1's share by the full alignment.
  EXPECT_EQ(constant(splits[1]), kTileArea / 4);
  EXPECT_EQ(constant(splits[0]), 4);
}

TEST(MakeSplitFactors, SpentReservationIsReleased) {
  slinky_globals globals;
  std::vector<slinky::expr> extents = {4096, 256};
  std::vector<slinky::expr> alignments = {16};
  std::vector<slinky::expr> splits = make_split_factors(
      globals, extents, kElementCost, /*given_splits=*/{}, /*loop_order=*/{},
      alignments);
  // Once dimension 0's split is chosen, its reservation must not keep
  // charging the area on top of the split itself.
  EXPECT_EQ(constant(splits[0]), 4096);
  EXPECT_EQ(constant(splits[1]), kTileArea / 4096);
}

}  // namespace
}  // namespace ynn
