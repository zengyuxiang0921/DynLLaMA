//
// Tokenizer test: round-trip + print ids for reference comparison.
//
// Build:
//   g++ -std=c++17 -O2 -static -I dynllama-headers/cpu -I dynllama-headers
//       tests/test-tokenizer.cpp -o test-tokenizer
// Run:
//   ./test-tokenizer qwen2.5-0.5b-instruct-q4_k_m.gguf
//

#include "meta.h"
#include "blocks/tokenizer.h"

#include <cstdio>
#include <string>
#include <vector>

static int g_fail = 0;
static void ok(bool c, const char * w) { std::printf("  %s %s\n", c ? "OK  " : "FAIL", w); if (!c) g_fail++; }

int main(int argc, char ** argv) {
    const std::string path = argc > 1 ? argv[1] : "qwen2.5-0.5b-instruct-q4_k_m.gguf";

    dynllama::gguf_meta m;
    if (!m.open(path)) { std::printf("open failed: %s\n", m.error().c_str()); return 2; }

    dynllama::bpe_tokenizer tok;
    std::string err;
    if (!tok.load(m, &err)) { std::printf("tokenizer load failed: %s\n", err.c_str()); return 2; }
    std::printf("vocab=%zu merges=%zu bos=%d eos=%d\n",
                tok.id_to_token.size(), tok.merge_rank.size(), tok.bos_id, tok.eos_id);

    struct sample { std::string text; std::vector<int> expect; };
    const std::vector<sample> samples = {
        {"Hello world", {9707, 1879}},
        {" The quick brown fox jumps over the lazy dog.",
         {576, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13}},
        {"def add(a, b):\n    return a + b\n",
         {750, 912, 2877, 11, 293, 982, 262, 470, 264, 488, 293, 198}},
        {"I'm don't we've 123 + 456 = 579",
         {40, 2776, 1513, 944, 582, 3003, 220, 16, 17, 18, 488, 220, 19, 20, 21, 284, 220, 20, 22, 24}},
        {"Mixed: caf\xc3\xa9 na\xc3\xafve \xe4\xbd\xa0\xe5\xa5\xbd",  // café naïve 你好
         {86433, 25, 51950, 94880, 586, 220, 108386}},
        {"line1\n\n  line2\ttab", {1056, 16, 271, 220, 1555, 17, 58149}},
    };

    for (const sample & s : samples) {
        std::vector<int> ids = tok.encode(s.text, /*add_special=*/false);
        std::string back = tok.decode(ids);
        ok(back == s.text, "round-trip");
        ok(ids == s.expect, "encode matches Qwen2 reference");
        if (ids != s.expect) {
            std::printf("    got    :");
            for (int id : ids) std::printf(" %d", id);
            std::printf("\n    expect :");
            for (int id : s.expect) std::printf(" %d", id);
            std::printf("\n");
        }
    }

    std::printf("\n%s (%d failure%s)\n", g_fail == 0 ? "PASSED" : "FAILED", g_fail, g_fail == 1 ? "" : "s");
    return g_fail == 0 ? 0 : 1;
}
