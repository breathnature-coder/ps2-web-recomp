# GT4 native rendering investigation

Goal: preserve EE recompilation and use GT4-specific render sequences to reduce
GS memory conversions and synchronization. No performance improvement is yet
established for a new native path or deko3d backend.

## Current foundation

- `recomp/vurecomp.cpp` generates C++ from captured VU1 code; `scripts/port.sh`
  includes that output in target builds. `vu_micro.cpp` selects compiled functions
  using code content and entry point, with an interpreter fallback. Actual GT4
  program coverage and ARM64 output still need auditing with game captures.
- `gpu.cpp` already has native rasterization and compute fallback. Native
  surfaces are stored to GS memory on conflicts and reloaded when needed.
- `lut_try()` recognizes colour-correction sequences. Keep this existing
  specialization; do not count its work as ordinary geometry batches.

## First instrumentation increment

`@env PS2_GS_PATH_STATS=1` enables bounded diagnostics in Switch args.txt.
Omit it or set it to `0` for normal runs. It does not require native rendering or
`--stats`; use existing timing stats alongside it when measuring performance.
Do not enable detailed trace logging during a performance comparison.

Every approximately five seconds, at submission boundaries, `[gs-path]` reports
interval totals, the eight families with the highest CPU preparation time, and
overflow totals. At most 256 families are retained per interval. This is a
first-seen bounded table, not a heavy-hitter estimator: if overflow is substantial,
the reported top eight cannot establish which families dominate the whole run.
All omitted rows are included in `total`; overflow is also included in `total`.
Values are interval totals, not rates; divide by `window_s` to obtain rates.

Family words (hex, in order): setup flags, framebuffer PSM, framebuffer mask,
depth PSM, packed modes, alpha reference, blend FIX, texture PSM (zero if unused),
primitive kind, mixed flag. Primitive kind is the renderer's internal record
kind, not the original GS PRIM register. A mixed batch is labelled by its first
family with mixed=1; its cost must not be assigned to the first family alone.

These keys are diagnostic groupings ONLY. They omit resource addresses, sizes,
palette contents, fog and vertex values and MUST NOT select a specialized
renderer or establish visual equivalence.

`cpu_prepare_ms` measures batch flushing on the CPU. It excludes the profiler's
family scan and aggregation, but instrumentation can still perturb cache and
wall-clock behaviour. It is NOT GPU time or total GS-thread time. `stores`,
`loads`, and feedback counts are operations scheduled inside that flush;
`feedback_source_pixels` counts unscaled copied source pixels, not bytes.
Native batch counts describe accepted routing, not final GPU draw counts.
Colour-correction specialization, uploads, display, readback, and stores outside
ordinary batch flushing are not attributed here. Compare with existing global
timing/transfer counters; do not mistake partial attribution for total traffic.
Short sessions ending before the reporting boundary may have no output.

## Surface conversion attribution

The same switch also emits `[gs-transfer]` in the same five-second window.
Unlike batch-family costs, these buckets include **all** calls to store/load
native surfaces, including calls outside ordinary batch flushing. Each operation
is counted once. This is not a count of all GPU copies, uploads or readbacks.

| Store cause | Meaning |
| --- | --- |
| 0 | Other/default synchronization, including copy/readback/display callers |
| 1 | Texture read by a native batch |
| 2 | Target of a compute batch |
| 3 | Texture read by a compute batch |
| 4 | Upload or CLUT operation |
| 5 | Overlapping native target |
| 6 | Native surface retirement/eviction |
| 7 | Explicit sync-all |

Load causes are a **decimal bit mask**: bits 0..5 use the corresponding write
causes above, bit 6 means newly allocated surface, and bit 7 means forced reload
by the existing PS2_NATIVE_SYNC diagnostic. A surface remembers all invalidating
causes until loaded, even across report windows. For example, mask 80 combines
initial allocation (64) and upload/CLUT (16). Its bytes belong to that combination
once, not independently to both causes. Reasons describe the pending stale span;
they do not partition that span by exact writer or identify the originating VU.

`source_pixels` is the unscaled scheduled rectangle area. `native_rect_bytes`
is its native RGBA8/Depth32Float footprint: area × scale² × 4. It is **not**
measured DRAM traffic, GS-format payload size, or GPU time. A repeated conversion
counts repeatedly. Page-aligned stale spans and store rectangles can include
untouched pixels, making these values useful for identifying amplification.

First compare large store/load buckets with GPU timings. A low draw count with
large conversion footprints can be a stronger candidate than a frequent cheap
draw. Retaining a target is safe only after proving all aliasing consumers can
read the retained representation or have a correctly ordered fallback.

## Next implementation gates

### Experimental sparse surface reloads

`@env PS2_NATIVE_SPARSE_LOAD=1` enables an opt-in resource-retention prototype.
The default (unset or `0`) keeps the previous bounding-interval reload policy.
This is an optimization of resource ownership in the existing renderer, not a
deko3d backend or a GT4 semantic draw replacement.

Each native surface retains up to four disjoint stale row spans. Reloads draw
only those spans, inside one native pass; untouched gaps keep their newer native
pixels. A fifth disjoint span collapses the list to its enclosing interval and
preserves any dirty native pixels before that expanded reload. Initial surfaces
and PS2_NATIVE_SYNC forced reloads still initialize the complete target.
Before a forced full reload, the opt-in path writes back retained dirty pixels
in the gaps; otherwise that diagnostic would destroy data the sparse path kept.
The conservative GS address-overlap checks and compatibility renderer remain.
Broad source dependency ranges can still invalidate the full interval, so this
does not guarantee savings for typical GT4 uploads or draws.

`[gpu] sparse loads` reports source pixels skipped relative to the enclosing
interval, and gap store decisions deferred. The latter is NOT net stores saved:
several decisions can concern the same dirty surface, which may later need a
writeback. Compare actual store totals and frame time separately. `loads` now
counts reload rectangles, not passes, when this option is enabled.

Local Mesa llvmpipe validation:

- Two targeted phases exercise disjoint top/bottom updates around dirty native
  pixels, and overflow to five spans. Both modes match the software GS reference.
  The prototype skips 143,360 source pixels; stores decrease from 6 to 5, while
  reload rectangles increase from 3 to 4. These are synthetic workload counts,
  not measured GT4 frame-rate improvements.
- Forty deterministic synthetic race scenes have byte-identical GS memory and
  display outputs between enabled and disabled modes (zero tolerance).
- The targeted fixture also passes at 2x resolution and with forced full reloads
  (`PS2_NATIVE_SYNC=1`); the latter includes the dirty-gap writeback regression.
- The bounded interval test runs 100,000 insertion steps with coverage checks,
  plus gap retention, collapse and boundary cases under ASan/UBSan.

Switch timing and GT4 scene coverage are still unverified. Keep this opt-in until
saved conversion work outweighs the extra draw and tracking costs on hardware.

1. Correlate VU code hashes and GIF sequences with costly families. Add transfer
   byte/range and ownership-transition attribution before choosing a final pass.
2. Candidate: ordinary textured opaque geometry. It is not yet established as
   the dominant cost. Validate exact state/resource guards and replay output
   against the current renderer before allowing a semantic shortcut.
3. Keep original ordering and compatibility fallback. The key opportunity is
   retaining native surfaces across a verified sequence, not merely replacing
   OpenGL API calls. Aliasing, framebuffer-as-texture and palette updates must
   invalidate or exit the shortcut correctly.
4. Bring up deko3d only after the sequence/resource contract is explicit. A
   fallback sharing the same backend/resource ownership is preferred; do not
   assume cross-API OpenGL/deko3d texture sharing or cheap round trips.
5. Validate identical scene, resolution and clocks: CPU submission, GPU frame
   time, conversions/bytes, waits, frame-time median/P95/P99 and image correctness.
   Desktop tests do not establish Switch speedups or 60 FPS.

## Local validation of this increment

- Bounded-family accounting test passes with AddressSanitizer and UBSan, and
  with warnings treated as errors. It checks relocation-insensitive grouping,
  significant state differences, saturation, overflow conservation and reset.
- Surface accounting tests cover scale-squared footprints, values exceeding
  32 bits, mixed invalidation buckets without double counting, and reset.
- Renderer syntax checks pass for both GL and WebGPU configurations.
- On Mesa llvmpipe, 12 deterministic synthetic race scenes produce byte-identical
  GS memory dumps with path statistics enabled and disabled. This comparison is
  between the two GPU runs; it does not assert equality with the software GS
  reference or prove GT4-specific visual correctness.
- After adding surface-conversion attribution, the enabled 12-scene dump also
  matches the pre-change disabled-statistics dump byte for byte.
- CMake is unavailable in the local environment; the GL comparison executable
  was built directly from the runtime sources with GCC. The new accounting test
  is registered with CTest for the existing CI workflow.
