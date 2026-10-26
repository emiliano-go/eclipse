// convert_targets: turn raw teacher logits into byte/EOT targets.
//
// Reads raw records from stdin (produced by dump_logits with the `observed`
// array), converts each position against the exported byte trie, and writes
// target records to stdout. No Python and no raw-logit persistence.
//
// Raw record (little-endian):
//   uint32 magic=0x474F4C42, n, vocab, first_pos, flags, doc_id
//   int32  ids[n]
//   int32  observed[n]        (-1 for the last token of a document)
//   logits: fp16 (flags=0) or fp32 (flags=1), n*vocab
//
// Target record:
//   uint32 magic=0x54415247, version=1, doc_id, first_pos, n_positions,
//          n_rows, vocab=260, flags=0
//   uint16 symbols[n_rows]
//   uint16 nexts[n_rows]
//   uint8  kinds[n_rows]
//   int32  doc_ids[n_rows]
//   float  targets[n_rows*260]

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

static constexpr uint32_t RAW_MAGIC = 0x474F4C42u;
static constexpr uint32_t TRIE_MAGIC = 0x54524945u;
static constexpr uint32_t TGT_MAGIC = 0x54415247u;
static constexpr int VOCAB_TGT = 260;
static constexpr int EOT = 256;
static constexpr int BOS = 257;

struct Trie {
    uint32_t n_nodes = 0, n_tokens = 0, n_logits = 0, max_len = 0;
    uint32_t n_valid = 0, root_hi = 0;
    std::vector<int32_t> node_lo, node_hi, exact_lo, exact_hi, child;
    std::vector<int32_t> token_len, order_ids;
    std::vector<uint8_t> token_bytes, is_special;
};

static void die(const char * msg) {
    std::fprintf(stderr, "convert_targets: %s\n", msg);
    std::exit(1);
}

static void read_all(FILE * f, void * dst, size_t bytes, const char * what) {
    if (bytes && std::fread(dst, 1, bytes, f) != bytes) {
        std::fprintf(stderr, "convert_targets: short read of %s (%zu bytes)\n", what, bytes);
        std::exit(1);
    }
}

static Trie load_trie(const char * path) {
    FILE * f = std::fopen(path, "rb");
    if (!f) { std::perror(path); std::exit(1); }
    uint32_t h[8];
    read_all(f, h, sizeof(h), "trie header");
    if (h[0] != TRIE_MAGIC || h[1] != 1) die("bad trie magic/version");
    Trie t;
    t.n_nodes = h[2];
    t.n_tokens = h[3];
    t.n_logits = h[4];
    t.max_len = h[5];
    t.n_valid = h[6];
    t.root_hi = h[7];
    if (!t.n_nodes || !t.n_tokens || !t.n_logits || !t.max_len) die("bad trie dimensions");
    if (t.root_hi != t.n_valid) die("root_hi != n_valid");

    auto load_i32 = [&](std::vector<int32_t> & v, size_t n, const char * name) {
        v.resize(n);
        read_all(f, v.data(), n * sizeof(int32_t), name);
    };
    auto load_u8 = [&](std::vector<uint8_t> & v, size_t n, const char * name) {
        v.resize(n);
        read_all(f, v.data(), n, name);
    };
    load_i32(t.node_lo, t.n_nodes, "node_lo");
    load_i32(t.node_hi, t.n_nodes, "node_hi");
    load_i32(t.exact_lo, t.n_nodes, "exact_lo");
    load_i32(t.exact_hi, t.n_nodes, "exact_hi");
    load_i32(t.child, (size_t)t.n_nodes * 256, "child");
    load_i32(t.token_len, t.n_tokens, "token_len");
    load_u8(t.token_bytes, (size_t)t.n_tokens * t.max_len, "token_bytes");
    load_i32(t.order_ids, t.n_valid, "order_ids");
    load_u8(t.is_special, t.n_tokens, "is_special");
    std::fclose(f);
    return t;
}

static inline float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while (!(mant & 0x400u)) { mant <<= 1; --exp; }
            mant &= 0x3ffu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

int main(int argc, char ** argv) {
    const char * trie_path = nullptr;
    int threads = 8;
    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char * n) -> const char * {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing %s\n", n); std::exit(1); }
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--trie")) trie_path = need("--trie");
        else if (!std::strcmp(argv[i], "--threads")) threads = std::atoi(need("--threads"));
        else { std::fprintf(stderr, "usage: %s --trie TRIE.bin [--threads N]\n", argv[0]); return 1; }
    }
    if (!trie_path) { std::fprintf(stderr, "usage: %s --trie TRIE.bin [--threads N]\n", argv[0]); return 1; }
    if (threads < 1) threads = 1;
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif

    Trie trie = load_trie(trie_path);
    std::fprintf(stderr, "trie: nodes=%u tokens=%u vocab=%u max_len=%u\n",
                 trie.n_nodes, trie.n_tokens, trie.n_logits, trie.max_len);

    std::vector<int32_t> ids, observed;
    std::vector<float> logits;
    std::vector<uint16_t> raw16;
    const int64_t t0 = 0; (void)t0;
    uint64_t total_positions = 0, total_rows = 0, records = 0;
    const clock_t start = std::clock();

    while (true) {
        uint32_t h[6];
        size_t got = std::fread(h, sizeof(uint32_t), 6, stdin);
        if (got == 0) break;
        if (got != 6) die("short raw header");
        if (h[0] != RAW_MAGIC) die("bad raw magic");
        const int n = (int32_t)h[1];
        const int vocab = (int32_t)h[2];
        const int32_t first_pos = (int32_t)h[3];
        const uint32_t flags = h[4];
        const int32_t doc_id = (int32_t)h[5];
        if (n <= 0 || vocab < (int)trie.n_tokens) die("bad raw dimensions");
        ids.resize(n);
        observed.resize(n);
        read_all(stdin, ids.data(), (size_t)n * 4, "ids");
        read_all(stdin, observed.data(), (size_t)n * 4, "observed");
        logits.resize((size_t)n * vocab);
        if (flags & 1u) {
            read_all(stdin, logits.data(), (size_t)n * vocab * 4, "fp32 logits");
        } else {
            raw16.resize((size_t)n * vocab);
            read_all(stdin, raw16.data(), (size_t)n * vocab * 2, "fp16 logits");
            for (size_t i = 0; i < raw16.size(); ++i) logits[i] = half_to_float(raw16[i]);
        }

        std::vector<int64_t> row_off((size_t)n + 1, 0);
        for (int i = 0; i < n; ++i) {
            int32_t tid = observed[i];
            int64_t rows = 0;
            if (tid >= 0) {
                if (tid >= (int32_t)trie.n_tokens || trie.is_special[tid]) die("bad observed token");
                int32_t len = trie.token_len[tid];
                if (len <= 0) die("observed token has no bytes");
                rows = len + 1;
            }
            row_off[i + 1] = row_off[i] + rows;
        }
        const int64_t n_rows = row_off[n];

        std::vector<uint16_t> symbols(n_rows);
        std::vector<uint16_t> nexts(n_rows);
        std::vector<uint8_t> kinds(n_rows);
        std::vector<int32_t> doc_ids(n_rows);
        std::vector<float> targets((size_t)n_rows * VOCAB_TGT);

#pragma omp parallel
        {
            std::vector<float> p(vocab);
            std::vector<double> c((size_t)trie.n_valid + 1);
            std::vector<float> row(VOCAB_TGT);
#pragma omp for schedule(dynamic, 1)
            for (int i = 0; i < n; ++i) {
                const int32_t tid = observed[i];
                if (tid < 0) continue;
                const int L = trie.token_len[tid];
                const float * lg = logits.data() + (size_t)i * vocab;

                float mx = -INFINITY;
                for (int j = 0; j < vocab; ++j) mx = std::max(mx, lg[j]);
                double sum = 0.0;
                for (int j = 0; j < vocab; ++j) {
                    float x = lg[j] - mx;
                    if (x < -60.0f) x = -60.0f;
                    p[j] = std::exp(x);
                    sum += p[j];
                }
                const float inv = (float)(1.0 / sum);
                for (int j = 0; j < vocab; ++j) p[j] *= inv;
                const float p_token = p[tid];

                c[0] = 0.0;
                for (uint32_t j = 0; j < trie.n_valid; ++j) {
                    c[j + 1] = c[j] + p[trie.order_ids[j]];
                }
                const double root_mass = c[trie.root_hi] - c[trie.node_lo[0]];

                const uint8_t * bytes = trie.token_bytes.data() + (size_t)tid * trie.max_len;
                int node = 0;
                const int64_t r0 = row_off[i];
                for (int d = 0; d <= L; ++d) {
                    const double mass = c[trie.node_hi[node]] - c[trie.node_lo[node]];
                    const double denom = (d == 0) ? 1.0 : std::max(mass, (double)p_token);
                    std::fill(row.begin(), row.end(), 0.0f);
                    const int32_t * ch = trie.child.data() + (size_t)node * 256;
                    for (int b = 0; b < 256; ++b) {
                        const int32_t u = ch[b];
                        if (u >= 0) {
                            const double cm = c[trie.node_hi[u]] - c[trie.node_lo[u]];
                            row[b] = (float)(cm / denom);
                        }
                    }
                    if (d < L) {
                        const int b = bytes[d];
                        const float floor = (float)(p_token / denom);
                        if (row[b] < floor) row[b] = floor;
                    }
                    const double exact = (d == L)
                        ? (double)p_token
                        : (c[trie.exact_hi[node]] - c[trie.exact_lo[node]]);
                    double eot = exact / denom;
                    if (d == 0) eot += 1.0 - root_mass;
                    row[EOT] = (float)eot;
                    for (int k = 0; k <= EOT; ++k) if (row[k] < 0.0f) row[k] = 0.0f;
                    double rs = 0.0;
                    for (int k = 0; k <= EOT; ++k) rs += row[k];
                    const float rinv = (float)(1.0 / std::max(rs, 1e-30));
                    for (int k = 0; k < VOCAB_TGT; ++k) row[k] *= rinv;
                    std::memcpy(targets.data() + (size_t)(r0 + d) * VOCAB_TGT, row.data(),
                                VOCAB_TGT * sizeof(float));

                    const int64_t rr = r0 + d;
                    if (d == 0) {
                        const bool first = (first_pos == 0 && i == 0);
                        symbols[rr] = first ? BOS : EOT;
                        kinds[rr] = first ? 2 : 1;
                        nexts[rr] = bytes[0];
                    } else {
                        symbols[rr] = bytes[d - 1];
                        kinds[rr] = 0;
                        nexts[rr] = (d < L) ? bytes[d] : EOT;
                    }
                    doc_ids[rr] = doc_id;
                    if (d < L) {
                        node = ch[bytes[d]];
                        if (node < 0) die("observed token path missing from trie");
                    }
                }
            }
        }

        const uint32_t oh[8] = {TGT_MAGIC, 1, (uint32_t)doc_id, (uint32_t)first_pos,
                                (uint32_t)n, (uint32_t)n_rows, VOCAB_TGT, 0};
        if (std::fwrite(oh, 4, 8, stdout) != 8) die("write header");
        if (n_rows) {
            if (std::fwrite(symbols.data(), 2, n_rows, stdout) != (size_t)n_rows) die("write symbols");
            if (std::fwrite(nexts.data(), 2, n_rows, stdout) != (size_t)n_rows) die("write nexts");
            if (std::fwrite(kinds.data(), 1, n_rows, stdout) != (size_t)n_rows) die("write kinds");
            if (std::fwrite(doc_ids.data(), 4, n_rows, stdout) != (size_t)n_rows) die("write doc_ids");
            if (std::fwrite(targets.data(), 4, (size_t)n_rows * VOCAB_TGT, stdout)
                != (size_t)n_rows * VOCAB_TGT) die("write targets");
        }
        std::fflush(stdout);
        total_positions += (uint64_t)n;
        total_rows += (uint64_t)n_rows;
        ++records;
        const double secs = (double)(std::clock() - start) / CLOCKS_PER_SEC;
        std::fprintf(stderr, "\rrecords=%llu positions=%llu rows=%llu %.0f pos/s",
                     (unsigned long long)records, (unsigned long long)total_positions,
                     (unsigned long long)total_rows, total_positions / std::max(secs, 1e-9));
    }
    std::fprintf(stderr, "\ndone: records=%llu positions=%llu rows=%llu\n",
                 (unsigned long long)records, (unsigned long long)total_positions,
                 (unsigned long long)total_rows);
    return 0;
}
