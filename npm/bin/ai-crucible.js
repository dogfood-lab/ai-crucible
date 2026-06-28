#!/usr/bin/env node
"use strict";

// Thin npm wrapper for the ai-crucible CLI. Pure JSON config — @mcptoolshop/npm-launcher derives
// the release-asset names from convention, downloads the platform binary from the ai-crucible
// GitHub Release, verifies its SHA256 against checksums-<version>.txt, caches it, and runs it
// with full arg passthrough.
//
// version + tag are read from THIS package's package.json at runtime — NEVER hardcoded — so the
// launcher can never drift from the published package. A hardcoded version silently fetched the
// previous release's binaries after a bump (the published 0.3.0 package shipped a bin pinned to
// 0.2.0, so `npx` ran 0.2.0 code); deriving it makes that drift structurally impossible.
//   binary:    ai-crucible-<version>-<os>-<arch>
//   checksums: checksums-<version>.txt
const { version } = require("../package.json");
process.env.MCPTOOLSHOP_LAUNCH_CONFIG = JSON.stringify({
  toolName: "ai-crucible",
  owner: "dogfood-lab",
  repo: "ai-crucible",
  version: version,
  tag: "v" + version,
});

require("@mcptoolshop/npm-launcher/bin/mcptoolshop-launch.js");
