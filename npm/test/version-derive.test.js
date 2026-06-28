"use strict";

// Regression test (ai-crucible HIGH finding): the npm launcher's version + tag MUST be derived
// from package.json at runtime, never hardcoded — else a published package ships a bin pinned to
// a stale release and `npx` runs the wrong (or missing) binaries. The published 0.3.0 package had
// a bin pinned to 0.2.0; this locks the derive so that drift is structurally impossible.
//
// Plain node + assert (the package has no test framework). Stubs @mcptoolshop/npm-launcher so
// requiring the bin does NOT attempt a real network download — we only want the side effect of
// the bin setting MCPTOOLSHOP_LAUNCH_CONFIG. Not in the published tarball (package.json `files`).

const assert = require("assert");
const Module = require("module");

const pkg = require("../package.json");

// Intercept the launcher require so the bin's `require(...launcher)` is a no-op here.
const origLoad = Module._load;
Module._load = function (request) {
  if (request.includes("@mcptoolshop/npm-launcher")) return {};
  return origLoad.apply(this, arguments);
};
try {
  require("../bin/ai-crucible.js");
} finally {
  Module._load = origLoad;
}

const cfg = JSON.parse(process.env.MCPTOOLSHOP_LAUNCH_CONFIG);
assert.strictEqual(cfg.version, pkg.version, "launch config version must equal package.json version");
assert.strictEqual(cfg.tag, "v" + pkg.version, "launch config tag must be v<package.json version>");
assert.strictEqual(cfg.toolName, "ai-crucible");
assert.strictEqual(cfg.owner, "dogfood-lab");
assert.strictEqual(cfg.repo, "ai-crucible");

console.log(`ok — launcher derives version ${cfg.version} / tag ${cfg.tag} from package.json`);
