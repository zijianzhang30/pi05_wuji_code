# Deployment continuation

Collection is complete: seven tasks x 50 accepted episodes = 350. No collection workers remained at GPU validation. `collection_gate.json` records acceptance reports, unique trajectory hashes and required artifact presence. This is not a new physics replay, broad generalization or independent smoothness certification.

## Completed

- CPU installer attempt 6 exited successfully. `environment_status.json` is now `gpu_ops_passed_weights_pending`. Do not restart installation.
- Official OpenPI SHA `215abfb217dbac7d5f1273282331b9b1866c0479`; submodules aloha `d1dc83afd89ded4379851257fe5d85632d31d5ec`, libero `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`.
- `installed_packages.txt` records the independent Python 3.11.14 environment. JAX 0.5.3 / jaxlib 0.5.3 / Flax 0.10.2 / Torch 2.7.1+cu126. Original pinned dependency versions and hashes remain unchanged. Only wheel transport URLs changed to individually probed mirrors; see `dependency_transport.json` and `upstream_uv.lock`.
- `gpu_ops_result.json` passed BF16 GEMM, float32 convolution and JIT gradient on one locked Blackwell GPU. cuDNN slow-operation alarms were autotuning, not a failure. The process has exited and released the lock. Do not upgrade dependencies merely because those alarms appeared.
- Official Paligemma tokenizer downloaded from generation 1711547605575873, exact size and CRC32C verified; cache `cache/openpi/big_vision/paligemma_tokenizer.model`, receipt `tokenizer_integrity.json`.
- `smoke_samples/` has seven real held-out NPZ/JPEG inputs, raw52 preserved, active26 padded to32 after normalization, missing wrist masks false. Statistics use only 315 training episodes; 35 held out. This is smoke input preparation, not full LeRobot conversion.

## Running: official weights

Inspect `weights_download_pid.json`, `weights_status.json`, actual process and `weights_ranges.log` before any restart. Current range downloader attempt 3 is PID 1308669. Attempt 2 was deliberately stopped only after verifying its exact command and process group, then acquiring its released lock. Repeated 240-second curl timeouts were discarding nearly complete 8-MB transfers. Attempt 3 resumes received prefixes within each range; verifies every returned Content-Range before appending, and retains all completed chunks. A mocked timeout/resume test verified correct offsets and byte-exact reconstruction. The actual final GCS CRC32C check remains mandatory. Attempt 2 script/log/status were archived as weights_ranges_attempt2.* / weights_ranges_attempt2_status.json. Attempt 1 exited with three failed ranges after 239/242 chunks completed for its first unfinished object. It was not still running when resumed. Completed files and chunks were retained. Attempt 2 relaxes low-speed cutoff to 8 KB/s for 45 seconds and retries five times, preventing premature rejection of slow valid transfers. Original failed status/log are archived as `weights_ranges_attempt1_*`.

Downloader uses an exclusive `weights_download.lock`, eight bounded workers, official generation-pinned GCS objects, Content-Range and size validation. The 29-object manifest totals 12,441,749,581 bytes. Files remain in `weights/pi05_base.partial` until checksums verify. Never run the old `download_weights.py` concurrently. Count completed chunks/files as well as status; an exceptional future may leave the counter stale while remaining futures finish.

## Next, in order

1. When downloader has exited with `downloaded_checksums_pending`, run independent `env/bin/python verify_weights.py` with Isaac environment variables removed. Check all 29 sizes + CRC32C, MD5 where supplied, record SHA256. Six composite objects lack MD5. Verify before publication to `weights/pi05_base`. No incomplete model loading. Verifier is prepared but has not run.
2. `model_forward.py` is prepared but has not run. Launch only via `python3 run_gpu_check.py model_forward.py`, holding the shared `dexterity_living/reports/process_manager/gpuN.lock` throughout. Wrapper rechecks collection, selects an idle GPU and removes Isaac environment variables. Do not duplicate an active wrapper.
3. The model test loads verified official base weights, runs ten-step sampling on seven held-out examples, checks each action output is finite and shaped 50x32, saves outputs and report. These are untrained normalized predictions, never robot commands. No ALOHA transforms.
4. If model test succeeds, save source/dependencies/weights/checksums/test evidence and isolated commands in a deployment report, sync small reports locally, deliver the verified result and limitations, then pause heartbeat `wuji-0-5`. No formal long training or autonomous closed-loop task success is authorized by this deployment heartbeat.

Do not modify Isaac environment, other users' jobs or source dataset. Ordinary advancing progress stays quiet; report only completed deployment, unresolved actionable errors or required user action.
