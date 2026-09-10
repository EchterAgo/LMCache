# Fork notice: upstream-incompatible features (`vllm-backport-0.13` branch)

This branch (`wtdcode/LMCache`, branch `vllm-backport-0.13`) carries features
that **diverge from upstream LMCache behavior**. It is the fork's
`vllm-backport` series re-based onto upstream `dev` at `c1276856`
(2026-09-10) and pairs with the vLLM fork branch `port-0.13`. If you come
from upstream, read this first; if you sync this branch onto upstream,
re-audit every item.

1. **L2 startup adoption** (`l2: adopt persisted objects ...`) — on server
   start, filesystem L2 adapters scan their directory and re-seed byte
   accounting + LRU from objects persisted by a previous run (oldest-first
   by mtime). Upstream restarts leave old objects servable but *unaccounted
   and unevictable*, so repeated restarts stack unaccounted generations
   until the disk fills. Default **on** (`"adopt_existing": true` in the
   fs_native adapter JSON); upstream has no such key (upstream PR #4571
   "restore fs_native cache state after restart" is the closest open
   proposal). Caveat: adoption trusts the on-disk objects' layout — changing
   `--kv-cache-dtype`, the attention backend / kernel block size, PP
   topology, or the connector's Mamba page view (see item 5) reuses
   same-keyed objects with a different payload layout (a pre-existing
   upstream hazard); wipe the directory when changing those, until the
   registration-time layout-fingerprint gate lands.

2. **Fail-closed rejection of spec decode + align-mode hybrid +
   `max_num_batched_tokens > block_size`** (`reject spec decode with
   multi-block prefill steps`) — upstream accepts this configuration; this
   fork rejects it at startup because multi-block prefill steps store
   recurrent-state chunks that deterministically corrupt later prefix hits
   (verified on Qwen3.8-Flash-Next FP8). Set `--max-num-batched-tokens`
   equal to the unified block size, or disable speculative decoding.
   Upstream PR #5004 ("Don't store MTP's speculative scratch block as a
   Mamba state") looks like the root-cause fix; re-evaluate this guard once
   it merges.

3. **Chunk-boundary state-checkpoint requirement for recurrent/SWA hybrids**
   (`require prefix-cache checkpoints on chunk boundaries`) — models with
   recurrent or sliding-window KV groups must run vLLM with
   `--prefix-cache-retention-interval` set to a divisor of the LMCache chunk
   size; the connector refuses to start otherwise. Upstream silently serves
   hits without a state checkpoint at the boundary.

4. **GLM-5.3-Flash / vLLM unified-layout support** (`re-view kernel-paged
   4-dim MLA/indexer pools`, `treat non-prefix-cacheable engine groups as
   non-positional`, `permute unified attention views`, `tokens_per_state`
   compression) — kernel-paged 4-dim MLA/indexer pools are re-viewed at
   logical-block granularity, unified `[B, H, N, C]` views are permuted
   into physical order (group edits now run for non-Mamba models and on
   leaf specs), `AttentionSpec.tokens_per_state > 1` is treated as slot
   compression, and connector-private scratch groups (`CircularBufferSpec`,
   `KpoolTailSpec`; `prefix_cacheable` / `participates_in_prefix_caching`
   is `False`) are excluded from registration, block-id slicing and
   capacity math. Upstream stores corrupt bytes for these pools, so **GLM
   cache directories written by upstream (or by the fork before the
   subpaged MLA edit) are invalid** — wipe them.

5. **Dropped in favour of upstream** — the fork's own `KVCacheLayout`
   resolution (`layer_view_order` probing), the connector-to-adapter
   layout-hint threading, the block-axis (dim-0 padded) `NL_X_NB_*_CS`
   formats, and the one-slot-per-block fallback for padded Mamba pages are
   all superseded by upstream `a0e4a4c2` ("vLLM LBHNC KVLayout metadata"),
   which maps `LBNHC`/`LBHNC`/`BLHNC`/`BLNHC` by name and re-views padded
   Mamba pages by `as_strided` tiling instead. The tiling changes the
   stored byte layout of Mamba-state objects relative to the old
   `vllm-backport` branch, so **wipe L2 directories of Mamba-hybrid models
   when moving from `vllm-backport` to this branch**. Upstream `ca2f4744`
   dropped the connector layout preference; the fork never had one.

Also carried (bug fixes, no config surface): the scheduler-side heartbeat
threads are actually started (`_heartbeats` guard), `REGISTER_KV_CACHE` /
`UNREGISTER_KV_CACHE` run on the affinity pool (declared `BLOCKING`) so
large-engine registration does not starve other workers' pings, prefetch
drops keys that fail read-lock acquisition from the hit set, and
contiguity validation ignores strides of size-1 dims.

Entries below marked upstream README content.

---

<div align="center">
  <p align="center">
    <img src="asset/logo.png" alt="lmcache logo" width="45%">
  </p>
  <h3 align="center">
    A KV Cache Management Layer for Scalable LLM Inference
  </h3>
    <hr width="78%">

  <h3 align="center">
    <a href="https://blog.lmcache.ai/">Blog</a> |
    <a href="https://docs.lmcache.ai/">Documentation</a> |
    <a href="https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-3zxjao8h0-lRfBfnLqbALOtLsWn2ITxA">Join Slack</a> |
    <a href="https://docs.lmcache.ai/community/meetings.html">Community Meeting</a> |
    <a href="https://github.com/LMCache/LMCache/issues/2923">Roadmap</a>
  </h3>

  [![GitHub Repo stars](https://img.shields.io/github/stars/LMCache/LMCache?style=flat&logo=github)](https://github.com/LMCache/LMCache/stargazers)
  [![PyPI](https://img.shields.io/pypi/v/lmcache)](https://pypi.org/project/lmcache/)
  [![PyPI - Downloads](https://img.shields.io/pypi/dm/lmcache)](https://pypi.org/project/lmcache/)
  [![GitHub commit activity](https://img.shields.io/github/commit-activity/w/LMCache/LMCache)](https://github.com/LMCache/LMCache/graphs/commit-activity)
  [![GitHub contributors](https://img.shields.io/github/contributors/LMCache/LMCache)](https://github.com/LMCache/LMCache/graphs/contributors)
  [![Good first issues](https://img.shields.io/github/issues/LMCache/LMCache/good%20first%20issue?label=good%20first%20issues&color=7057ff)](https://github.com/LMCache/LMCache/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
  [![Slack](https://img.shields.io/badge/Slack-Join%20us-4A154B?logo=slack)](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-3zxjao8h0-lRfBfnLqbALOtLsWn2ITxA)
  [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/LMCache/LMCache/)

  ⭐ **If LMCache helps you serve LLMs faster and cheaper, [give us a star](https://github.com/LMCache/LMCache) — it helps more teams discover the project.**

</div>

## Updates
- [2026/05] 🔥 Agentic workload benchmark on AMD MI300X ([blog](https://blog.lmcache.ai/en/2026/05/12/benchmarking-lmcache-for-multi-turn-agentic-workloads-on-amd-mi300x/)).
- [2026/04] 🔥 LMCache's new multiprocess (MP) architecture release ([blog](https://blog.lmcache.ai/en/2026/04/03/lmcaches-new-architecture-boosts-moe-inference-performance-by-10x/)).
- [2026/03] LMCache at GTC 2026 ([post](https://www.linkedin.com/posts/lmcache-lab_llm-opensource-nvidiagtc-activity-7442721875664826369-pMAu?utm_source=share&utm_medium=member_desktop&rcm=ACoAADkIIvQBTyG53kXXX70OZdE5rhpllYQqmIA)).
- [2026/01] LMCache multi-node P2P CPU memory sharing, from experimental feature to production ([blog](https://blog.lmcache.ai/en/2026/01/21/p2p-1/)).

<details>
<summary>More</summary>

- [2025/11] LMCache x CoreWeave accelerate efficient LLM inference for Cohere ([blog](https://blog.lmcache.ai/en/2025/10/29/breaking-the-memory-barrier-how-lmcache-and-coreweave-power-efficient-llm-inference-for-cohere/)).
- [2025/10] LMCache joins the PyTorch Foundation and Tensormesh unveiled ([blog](https://blog.lmcache.ai/en/2025/10/31/tensormesh-unveiled-and-lmcache-joins-the-pytorch-foundation/), [PyTorch](https://pytorch.org/blog/lmcache-joins-pytorch-ecosystem/)).
- [2025/09] NVIDIA Dynamo integrates LMCache, accelerating LLM inference ([blog](https://blog.lmcache.ai/en/2025/09/18/nvidia-dynamo-integrates-lmcache-accelerating-llm-inference/)).
- [2025/08] 🎉 LMCache hits 5,000+ GitHub stars ([blog](https://blog.lmcache.ai/en/2025/08/28/%f0%9f%8e%89-lmcache-hits-5000-github-stars-thank-you-community/)).
- [2025/08] LMCache supports gpt-oss (20B/120B) on day 1 ([blog](https://blog.lmcache.ai/en/2025/08/05/lmcache-supports-gpt-oss-20b-120b-on-day-1/)).
- [2025/07] Get faster LLM inference and cheaper responses with LMCache and Redis ([Redis blog](https://redis.io/blog/get-faster-llm-inference-and-cheaper-responses-with-lmcache-and-redis/)).
- [2025/07] LMCache extends its turbo-boost to multimodal models in vLLM V1 ([blog](https://blog.lmcache.ai/en/2025/07/03/lmcache-extends-its-turbo-boost-to-multimodal-models-in-vllm-v1/)).
- [2025/06] LLM Production Stack goes cross-hardware: AMD, Arm and Ascend ([blog](https://blog.lmcache.ai/en/2025/06/20/llm-production-stack-goes-cross-hardware-ascend-arm-and-amd-support-incoming/)).

</details>

## About

LMCache is a **KV cache management layer** for LLM inference. It turns KV cache from a temporary state into reusable *AI-native knowledge* that can be *stored* persistently, *reused* across multiple serving engines, *monitored* with an observability stack, and *transformed* for better generation quality. As a result, LMCache **reduces TTFT** (time-to-first-token) and **improves throughput**, especially for long-context agentic, multi-turn conversation, and knowledge-augmented workloads (e.g., RAG).

LMCache is **vendor-neutral**. It can be used as a KV cache layer for a range of mainstream open-source serving engines, inference frameworks, hardware vendors, storage systems, and infrastructure providers. The vendor neutrality allows users to freely switch between serving engines and storage vendors, while reusing the stored KV caches.

<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="asset/deployment_modes_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="asset/deployment_modes_light.png">
  <img alt="LMCache Deployment Modes" src="asset/deployment_modes_light.png">
</picture>
</p>

### Key features

- **Engine-independent deployment**: LMCache, as a standalone daemon process, manages KV cache independently from the inference engine process, so that KV cache will not be lost even if the inference engine crashes (i.e., no fate-sharing with engines).

- **Persistent, tiered KV cache offloading and reuse**: Move KV caches out of GPU memory into a tiered storage hierarchy spanning CPU memory, local storage, and remote backends, enabling reuse across requests, sessions, and engine instances to reduce repeated prefill computation and improve TTFT.

- **Production-level KV cache observability**: LMCache provides a rich set of KV cache observability metrics, including typical Kubernetes metrics (health monitoring, performance diagnostics), KV-cache-specific metrics (request-level and token-level prefix cache hits, lifecycle, request-level KV cache performance), management metrics (user-specific usage), and more.

- **Pluggable storage and transport backends**: Easily integrate remote storage and KV transfer backends through a unified interface, enabling KV cache offloading and sharing across storage providers. Through this interface, LMCache supports storage backends including CPU RAM, local disk (SSD), Redis/Valkey, Mooncake, InfiniStore, S3-compatible object storage, NIXL, and GDS.

- **Non-prefix KV reuse**: Extend KV reuse beyond prefix caching by reusing cached KV blocks at any position in the prompt. This leverages CacheBlend to selectively recompute tokens for quality recovery.

- **PD disaggregation and KV transfer**: Support KV cache transfer from prefill workers to decode workers over NVLink, RDMA, or TCP through transport layers such as NIXL.

- **Pluggable KV transformation**: A simple interface for researchers to write compression, token dropping, and custom serialization through a flexible SERDE interface.

LMCache is becoming an integral layer in the LLM inference *ecosystem*, with *community*-driven integration with serving engines, inference frameworks, hardware vendors, storage systems, and infrastructure providers:

<p align="center">
  <img src="asset/ecosystem.png" alt="LMCache ecosystem">
</p>

## Getting Started

To use LMCache, simply install `lmcache` from your package manager, e.g. pip:
```bash
pip install lmcache
```

For more setup options and examples, see:
- [Installation](https://docs.lmcache.ai/getting_started/installation.html)
- [Quickstart](https://docs.lmcache.ai/getting_started/quickstart.html)
- [LMCache Recipes](https://docs.lmcache.ai/recipes/index.html)
- [CLI Reference](https://docs.lmcache.ai/cli/index.html)
- [Benchmarking Guide](https://docs.lmcache.ai/getting_started/benchmarking.html)
- [Production Deployment](https://docs.lmcache.ai/mp/deployment.html)

## Contributing
We welcome and value contributions and collaborations. Join us in improving LMCache. Check out the [Contributing Guide](https://docs.lmcache.ai/developer_guide/contributing.html) or join our [Slack community](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-3zxjao8h0-lRfBfnLqbALOtLsWn2ITxA) to get started.

## Adoption and Partnerships
LMCache has a growing community of developers, researchers, industry adopters, and partners building the next generation of efficient LLM inference systems.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="asset/partner_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="asset/partner_light.png">
    <img alt="LMCache Adoption and Partnerships" src="asset/partner_light.png">
  </picture>
</p>

As an independent open-source project, LMCache is becoming the de-facto standard for KV Cache management in LLM inference. Its continued development and community work are supported in part by [Tensormesh](https://www.tensormesh.ai/).

## Citation

LMCache builds on research in KV cache management, including cache reuse, offloading, compression, and serving optimization. If you use LMCache in your research, please cite the LMCache paper and related work.

~~~bibtex
@article{cheng2025lmcache,
  title={LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference},
  author={Cheng, Yihua and Liu, Yuhan and Yao, Jiayi and An, Yuwei and Chen, Xiaokun and Feng, Shaoting and Huang, Yuyang and Shen, Samuel and Du, Kuntai and Jiang, Junchen},
  journal={arXiv preprint arXiv:2510.09665},
  year={2025}
}
~~~

<details>
<summary>Related papers</summary>

~~~bibtex
@inproceedings{liu2024cachegen,
  title={Cachegen: Kv cache compression and streaming for fast large language model serving},
  author={Liu, Yuhan and Li, Hanchen and Cheng, Yihua and Ray, Siddhant and Huang, Yuyang and Zhang, Qizheng and Du, Kuntai and Yao, Jiayi and Lu, Shan and Ananthanarayanan, Ganesh and others},
  booktitle={Proceedings of the ACM SIGCOMM 2024 Conference},
  pages={38--56},
  year={2024}
}

@inproceedings{yao2025cacheblend,
  title={Cacheblend: Fast large language model serving for rag with cached knowledge fusion},
  author={Yao, Jiayi and Li, Hanchen and Liu, Yuhan and Ray, Siddhant and Cheng, Yihua and Zhang, Qizheng and Du, Kuntai and Lu, Shan and Jiang, Junchen},
  booktitle={Proceedings of the twentieth European conference on computer systems},
  pages={94--109},
  year={2025}
}
~~~

</details>

## License

The LMCache codebase is licensed under Apache License 2.0. See the [LICENSE](LICENSE) file for details.
