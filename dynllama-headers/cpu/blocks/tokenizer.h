//
// Byte-level BPE tokenizer (GPT-2 / Qwen2 family), loaded from gguf metadata.
//
// gguf keys used:
//   tokenizer.ggml.model  = "gpt2"
//   tokenizer.ggml.tokens = [vocab]   (byte-encoded token strings)
//   tokenizer.ggml.merges = ["A B"]   (ranked merge rules)
//   tokenizer.ggml.{bos,eos,padding}_token_id, .add_bos_token
//
// encode():  text -> token ids   (pre-tokenize -> byte-encode -> BPE merge -> lookup)
// decode():  token ids -> text   (concat token bytes -> byte-decode)
//
// decode(encode(x)) == x is guaranteed for any input (byte-level coverage).
// The Qwen2 pre-tokenizer is approximated (full Unicode \p{L}/\p{N} categories
// are reduced to ASCII + "non-ASCII is a letter"); this only affects which
// merges fire for exotic Unicode, never round-trip correctness.
//

#ifndef DYNLLAMA_TOKENIZER_H
#define DYNLLAMA_TOKENIZER_H

#include "meta.h"

#include <climits>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace dynllama {

struct bpe_tokenizer {
    std::vector<std::string>                id_to_token;
    std::unordered_map<std::string, int>    token_to_id;
    std::unordered_map<std::string, int>    merge_rank; // key: "left right"

    std::string                             byte_to_sym[256]; // GPT-2 byte -> utf8(codepoint)
    std::unordered_map<uint32_t, uint8_t>   cp_to_byte;       // codepoint -> original byte

    int  bos_id = -1, eos_id = -1, pad_id = -1;
    bool add_bos = false;

    // ---- utf8 helpers ----
    static void utf8_append(uint32_t cp, std::string & out) {
        if (cp < 0x80) {
            out.push_back((char) cp);
        } else if (cp < 0x800) {
            out.push_back((char) (0xC0 | (cp >> 6)));
            out.push_back((char) (0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back((char) (0xE0 | (cp >> 12)));
            out.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char) (0x80 | (cp & 0x3F)));
        } else {
            out.push_back((char) (0xF0 | (cp >> 18)));
            out.push_back((char) (0x80 | ((cp >> 12) & 0x3F)));
            out.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char) (0x80 | (cp & 0x3F)));
        }
    }

    struct cp_span { uint32_t cp; size_t off; size_t len; };

    static std::vector<cp_span> utf8_decode(const std::string & s) {
        std::vector<cp_span> out;
        size_t i = 0, n = s.size();
        while (i < n) {
            uint8_t c = (uint8_t) s[i];
            uint32_t cp; size_t len;
            if (c < 0x80)        { cp = c; len = 1; }
            else if ((c >> 5) == 0x6 && i + 1 < n) { cp = ((c & 0x1F) << 6) | ((uint8_t) s[i+1] & 0x3F); len = 2; }
            else if ((c >> 4) == 0xE && i + 2 < n) { cp = ((c & 0x0F) << 12) | (((uint8_t) s[i+1] & 0x3F) << 6) | ((uint8_t) s[i+2] & 0x3F); len = 3; }
            else if ((c >> 3) == 0x1E && i + 3 < n){ cp = ((c & 0x07) << 18) | (((uint8_t) s[i+1] & 0x3F) << 12) | (((uint8_t) s[i+2] & 0x3F) << 6) | ((uint8_t) s[i+3] & 0x3F); len = 4; }
            else { cp = c; len = 1; } // invalid byte: treat as latin1
            out.push_back({cp, i, len});
            i += len;
        }
        return out;
    }

    void build_byte_maps() {
        int n = 0;
        for (int b = 0; b < 256; b++) {
            const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            const uint32_t cp = printable ? (uint32_t) b : (uint32_t) (256 + n++);
            std::string s; utf8_append(cp, s);
            byte_to_sym[b] = s;
            cp_to_byte[cp] = (uint8_t) b;
        }
    }

    bool load(const gguf_meta & m, std::string * err = nullptr) {
        if (m.get_str("tokenizer.ggml.model") != "gpt2") {
            if (err) *err = "tokenizer model is not gpt2 (byte-level BPE)";
            return false;
        }
        const auto * toks = m.get_str_array("tokenizer.ggml.tokens");
        const auto * mgs  = m.get_str_array("tokenizer.ggml.merges");
        if (!toks) { if (err) *err = "missing tokenizer.ggml.tokens"; return false; }

        build_byte_maps();

        id_to_token = *toks;
        token_to_id.reserve(id_to_token.size() * 2);
        for (int i = 0; i < (int) id_to_token.size(); i++) token_to_id[id_to_token[i]] = i;

        if (mgs) {
            merge_rank.reserve(mgs->size() * 2);
            for (int i = 0; i < (int) mgs->size(); i++) merge_rank[(*mgs)[i]] = i;
        }

        bos_id  = (int) m.get_i32("tokenizer.ggml.bos_token_id", -1);
        eos_id  = (int) m.get_i32("tokenizer.ggml.eos_token_id", -1);
        pad_id  = (int) m.get_i32("tokenizer.ggml.padding_token_id", -1);
        add_bos = m.get_bool("tokenizer.ggml.add_bos_token", false);
        return true;
    }

    // ---- pre-tokenization (approximate Qwen2 regex) ----
    static bool is_letter(uint32_t c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c >= 0x80; }
    static bool is_digit (uint32_t c) { return c >= '0' && c <= '9'; }
    static bool is_space (uint32_t c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }
    static bool is_nl    (uint32_t c) { return c == '\n' || c == '\r'; }

    // partition text into raw-byte pieces covering every byte exactly once
    std::vector<std::string> pretokenize(const std::string & text) const {
        const std::vector<cp_span> cps = utf8_decode(text);
        const int N = (int) cps.size();
        std::vector<std::string> out;

        auto emit = [&](int i, int j) {
            const size_t a = cps[i].off;
            const size_t b = cps[j - 1].off + cps[j - 1].len;
            out.push_back(text.substr(a, b - a));
        };
        auto lower = [](uint32_t c) -> uint32_t { return (c >= 'A' && c <= 'Z') ? c + 32 : c; };

        int i = 0;
        while (i < N) {
            const uint32_t c = cps[i].cp;

            // contractions: 's 't 're 've 'm 'll 'd
            if (c == '\'' && i + 1 < N) {
                uint32_t n1 = lower(cps[i + 1].cp);
                if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') { emit(i, i + 2); i += 2; continue; }
                if (i + 2 < N) {
                    uint32_t n2 = lower(cps[i + 2].cp);
                    if ((n1 == 'r' && n2 == 'e') || (n1 == 'v' && n2 == 'e') || (n1 == 'l' && n2 == 'l')) { emit(i, i + 3); i += 3; continue; }
                }
            }

            // optional single non-(letter/digit/newline) char, then letters: " ?word" / "(word"
            if (!is_letter(c) && !is_digit(c) && !is_nl(c) && i + 1 < N && is_letter(cps[i + 1].cp)) {
                int j = i + 1;
                while (j < N && is_letter(cps[j].cp)) j++;
                emit(i, j); i = j; continue;
            }
            if (is_letter(c)) {
                int j = i;
                while (j < N && is_letter(cps[j].cp)) j++;
                emit(i, j); i = j; continue;
            }

            // single digit (Qwen splits numbers per-digit)
            if (is_digit(c)) { emit(i, i + 1); i += 1; continue; }

            // optional leading space, then punctuation run, then trailing newlines
            if (!is_space(c) || (c == ' ' && i + 1 < N && !is_space(cps[i + 1].cp) && !is_letter(cps[i + 1].cp) && !is_digit(cps[i + 1].cp))) {
                int j = i;
                if (c == ' ') j++; // optional leading space
                int k = j;
                while (k < N && !is_space(cps[k].cp) && !is_letter(cps[k].cp) && !is_digit(cps[k].cp)) k++;
                if (k > j) {
                    while (k < N && is_nl(cps[k].cp)) k++;
                    emit(i, k); i = k; continue;
                }
            }

            // whitespace run. Qwen2:
            //   `\s*[\r\n]+`  - up to and including the last newline
            //   `\s+(?!\S)`   - a run followed by a non-space leaves its LAST
            //                   space for the next word/punct (" return", not "return")
            if (is_space(c)) {
                int j = i;
                while (j < N && is_space(cps[j].cp)) j++;
                int last_nl = -1;
                for (int k = i; k < j; k++) if (is_nl(cps[k].cp)) last_nl = k;
                if (last_nl >= 0) { emit(i, last_nl + 1); i = last_nl + 1; continue; }
                if (j < N && !is_space(cps[j].cp) && (j - i) > 1) { emit(i, j - 1); i = j - 1; continue; }
                emit(i, j); i = j; continue;
            }

            // fallback: single codepoint (guarantees progress + full coverage)
            emit(i, i + 1); i += 1;
        }
        return out;
    }

    // ---- BPE on one piece's byte-encoded symbols ----
    void bpe(std::vector<std::string> & syms) const {
        while (syms.size() > 1) {
            int best_rank = INT_MAX, best_i = -1;
            for (int i = 0; i + 1 < (int) syms.size(); i++) {
                auto it = merge_rank.find(syms[i] + " " + syms[i + 1]);
                if (it != merge_rank.end() && it->second < best_rank) {
                    best_rank = it->second; best_i = i;
                }
            }
            if (best_i < 0) break;
            syms[best_i] += syms[best_i + 1];
            syms.erase(syms.begin() + best_i + 1);
        }
    }

    std::vector<int> encode(const std::string & text, bool add_special = true) const {
        std::vector<int> ids;
        if (add_special && add_bos && bos_id >= 0) ids.push_back(bos_id);

        for (const std::string & piece : pretokenize(text)) {
            std::vector<std::string> syms;
            syms.reserve(piece.size());
            for (unsigned char b : piece) syms.push_back(byte_to_sym[b]);
            bpe(syms);
            for (const std::string & s : syms) {
                auto it = token_to_id.find(s);
                if (it != token_to_id.end()) {
                    ids.push_back(it->second);
                } else {
                    // should not happen (single-byte tokens always exist); fall back per byte
                    for (unsigned char b : s) {
                        auto bit = token_to_id.find(byte_to_sym[b]);
                        if (bit != token_to_id.end()) ids.push_back(bit->second);
                    }
                }
            }
        }
        return ids;
    }

    std::string decode(const std::vector<int> & ids) const {
        std::string enc;
        for (int id : ids) {
            if (id < 0 || id >= (int) id_to_token.size()) continue;
            enc += id_to_token[id];
        }
        std::string out;
        for (const cp_span & s : utf8_decode(enc)) {
            auto it = cp_to_byte.find(s.cp);
            if (it != cp_to_byte.end()) out.push_back((char) it->second);
            else utf8_append(s.cp, out); // not a byte-encoded cp (e.g. special token): keep literal
        }
        return out;
    }
};

} // namespace dynllama

#endif // DYNLLAMA_TOKENIZER_H
