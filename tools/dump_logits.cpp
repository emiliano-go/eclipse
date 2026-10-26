// dump_logits: stream per-position teacher logits from a GGUF model.
//
// Reads int32 token ids (already tokenized by the HF tokenizer), decodes
// fixed-size chunks with a fresh memory state, and writes one record per
// chunk to stdout:
//
//   uint32 magic     0x474F4C42 ("BLOG" little-endian)
//   uint32 n_tokens
//   uint32 n_vocab
//   uint32 first_pos
//   uint32 flags     0 = fp16, 1 = fp32
//   int32  ids[n_tokens]
//   logits[n_tokens][n_vocab]   (fp16 or fp32, row-major)
//
// The Python driver converts each record to byte/EOT targets immediately;
// raw logits are never persisted.

#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static constexpr uint32_t MAGIC = 0x474F4C42u;

static void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s -m MODEL.gguf --ids IDS.i32 [--doc-starts FILE] [--ctx N] [--batch N] [--ngl N] [--out FILE] [--fp32] [--limit N]\n"
        "       %s -m MODEL.gguf --tokenize TEXT [--out IDS.i32]\n", argv0, argv0);
}

static std::string read_file(const char * path) {
    FILE * f = std::fopen(path, "rb");
    if (!f) { std::perror(path); std::exit(1); }
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::string s(static_cast<size_t>(sz), '\0');
    if (sz > 0 && std::fread(s.data(), 1, static_cast<size_t>(sz), f) != static_cast<size_t>(sz)) {
        std::fprintf(stderr, "short read: %s\n", path);
        std::exit(1);
    }
    std::fclose(f);
    return s;
}

static std::vector<int32_t> read_ids(const char * path) {
    std::string raw = read_file(path);
    if (raw.size() % 4 != 0) {
        std::fprintf(stderr, "ids file size is not a multiple of 4\n");
        std::exit(1);
    }
    std::vector<int32_t> ids(raw.size() / 4);
    std::memcpy(ids.data(), raw.data(), raw.size());
    return ids;
}

static void write_all(FILE * f, const void * data, size_t bytes) {
    if (bytes && std::fwrite(data, 1, bytes, f) != bytes) {
        std::perror("fwrite");
        std::exit(1);
    }
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * ids_path = nullptr;
    const char * tokenize_path = nullptr;
    const char * doc_starts_path = nullptr;
    const char * out_path = nullptr;
    int32_t ctx_size = 2048;
    int32_t n_batch = 256;
    int32_t limit = -1;
    int32_t ngl = 99;
    bool fp32 = false;

    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char * name) -> const char * {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", name); std::exit(1); }
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "-m") || !std::strcmp(argv[i], "--model")) model_path = need("--model");
        else if (!std::strcmp(argv[i], "--ids")) ids_path = need("--ids");
        else if (!std::strcmp(argv[i], "--tokenize")) tokenize_path = need("--tokenize");
        else if (!std::strcmp(argv[i], "--doc-starts")) doc_starts_path = need("--doc-starts");
        else if (!std::strcmp(argv[i], "--out")) out_path = need("--out");
        else if (!std::strcmp(argv[i], "--ctx")) ctx_size = std::atoi(need("--ctx"));
        else if (!std::strcmp(argv[i], "--batch")) n_batch = std::atoi(need("--batch"));
        else if (!std::strcmp(argv[i], "--limit")) limit = std::atoi(need("--limit"));
        else if (!std::strcmp(argv[i], "--ngl")) ngl = std::atoi(need("--ngl"));
        else if (!std::strcmp(argv[i], "--fp32")) fp32 = true;
        else { usage(argv[0]); return 1; }
    }
    if (!model_path || (!ids_path && !tokenize_path)) {
        usage(argv[0]);
        return 1;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        std::fprintf(stderr, "failed to load model: %s\n", model_path);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    if (tokenize_path) {
        std::string text = read_file(tokenize_path);
        std::vector<llama_token> tokens(text.size() + 16);
        int32_t n = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                                   tokens.data(), static_cast<int32_t>(tokens.size()), false, true);
        if (n < 0) {
            tokens.resize(static_cast<size_t>(-n));
            n = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                               tokens.data(), static_cast<int32_t>(tokens.size()), false, true);
        }
        if (n < 0) { std::fprintf(stderr, "tokenize failed\n"); return 1; }
        FILE * out = out_path ? std::fopen(out_path, "wb") : stdout;
        if (!out) { std::perror(out_path); return 1; }
        write_all(out, tokens.data(), static_cast<size_t>(n) * sizeof(llama_token));
        if (out_path) std::fclose(out);
        std::fprintf(stderr, "tokenized %d tokens, vocab=%d\n", n, n_vocab);
        llama_model_free(model);
        return 0;
    }

    std::vector<int32_t> ids = read_ids(ids_path);
    if (limit >= 0 && static_cast<size_t>(limit) < ids.size()) ids.resize(static_cast<size_t>(limit));
    if (ids.empty()) { std::fprintf(stderr, "no ids\n"); return 1; }

    std::vector<int32_t> doc_starts;
    if (doc_starts_path) {
        doc_starts = read_ids(doc_starts_path);
        if (doc_starts.empty() || doc_starts.front() != 0) {
            std::fprintf(stderr, "doc-starts must begin at 0\n");
            return 1;
        }
        for (size_t i = 1; i < doc_starts.size(); ++i) {
            if (doc_starts[i] < doc_starts[i - 1] || doc_starts[i] > static_cast<int32_t>(ids.size())) {
                std::fprintf(stderr, "doc-starts must be sorted and within ids\n");
                return 1;
            }
        }
        doc_starts.push_back(static_cast<int32_t>(ids.size()));
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = static_cast<uint32_t>(ctx_size);
    cparams.n_batch = static_cast<uint32_t>(n_batch);
    cparams.n_ubatch = static_cast<uint32_t>(n_batch);
    cparams.n_seq_max = 1;
    cparams.n_threads = 8;
    cparams.n_threads_batch = 8;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    cparams.no_perf = true;
    cparams.op_offload = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { std::fprintf(stderr, "failed to create context\n"); return 1; }

    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    FILE * out = out_path ? std::fopen(out_path, "wb") : stdout;
    if (!out) { std::perror(out_path); return 1; }

    std::vector<uint16_t> buf16;
    std::vector<float> buf32;
    const int64_t t0 = llama_time_us();
    size_t done = 0;

    // Decode each document (or the whole stream) in segments of at most
    // n_ctx tokens. Memory is cleared at every segment start and at every
    // document start, but not between sub-batches, so tokens see their
    // preceding context within the current document/segment.
    auto decode_range = [&](size_t doc_begin, size_t doc_end, int32_t doc_id) {
        for (size_t seg = doc_begin; seg < doc_end; seg += static_cast<size_t>(ctx_size)) {
            const size_t seg_end = std::min(seg + static_cast<size_t>(ctx_size), doc_end);
            llama_memory_clear(llama_get_memory(ctx), true);

            for (size_t sub = seg; sub < seg_end; sub += static_cast<size_t>(n_batch)) {
                const int32_t n = static_cast<int32_t>(
                    std::min<size_t>(static_cast<size_t>(n_batch), seg_end - sub));
                batch.n_tokens = n;
                for (int32_t i = 0; i < n; ++i) {
                    batch.token[i] = ids[sub + static_cast<size_t>(i)];
                    batch.pos[i] = static_cast<llama_pos>(sub - seg) + i;
                    batch.n_seq_id[i] = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i] = 1;
                }

                if (llama_decode(ctx, batch) != 0) {
                    std::fprintf(stderr, "llama_decode failed at %zu\n", sub);
                    return false;
                }

                const uint32_t flags = fp32 ? 1u : 0u;
                const uint32_t hdr[6] = {MAGIC, static_cast<uint32_t>(n),
                                         static_cast<uint32_t>(n_vocab),
                                         static_cast<uint32_t>(sub - doc_begin), flags,
                                         static_cast<uint32_t>(doc_id)};
                write_all(out, hdr, sizeof(hdr));
                write_all(out, ids.data() + sub, static_cast<size_t>(n) * sizeof(int32_t));

                if (fp32) {
                    buf32.resize(static_cast<size_t>(n) * static_cast<size_t>(n_vocab));
                    for (int32_t i = 0; i < n; ++i) {
                        const float * row = llama_get_logits_ith(ctx, i);
                        std::memcpy(buf32.data() + static_cast<size_t>(i) * n_vocab, row,
                                    static_cast<size_t>(n_vocab) * sizeof(float));
                    }
                    write_all(out, buf32.data(), buf32.size() * sizeof(float));
                } else {
                    buf16.resize(static_cast<size_t>(n) * static_cast<size_t>(n_vocab));
                    for (int32_t i = 0; i < n; ++i) {
                        const float * row = llama_get_logits_ith(ctx, i);
                        uint16_t * dst = buf16.data() + static_cast<size_t>(i) * n_vocab;
                        for (int j = 0; j < n_vocab; ++j) dst[j] = ggml_fp32_to_fp16(row[j]);
                    }
                    write_all(out, buf16.data(), buf16.size() * sizeof(uint16_t));
                }
                std::fflush(out);

                done += static_cast<size_t>(n);
                const double secs = static_cast<double>(llama_time_us() - t0) / 1e6;
                std::fprintf(stderr, "\rdoc=%d %d/%zu tokens  %.1f tok/s  ctx=%d batch=%d",
                             doc_id, n, done, static_cast<double>(done) / std::max(secs, 1e-9),
                             ctx_size, n_batch);
            }
        }
        return true;
    };

    bool ok = true;
    if (!doc_starts.empty()) {
        for (size_t i = 0; i + 1 < doc_starts.size() && ok; ++i) {
            const size_t begin = static_cast<size_t>(doc_starts[i]);
            const size_t end = static_cast<size_t>(doc_starts[i + 1]);
            if (end > begin) ok = decode_range(begin, end, static_cast<int32_t>(i));
        }
    } else {
        ok = decode_range(0, ids.size(), 0);
    }
    if (!ok) return 1;
    std::fprintf(stderr, "\n");

    if (out_path) std::fclose(out);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    std::fprintf(stderr, "done: %zu tokens, %d vocab, %.1fs\n", done, n_vocab,
                 static_cast<double>(llama_time_us() - t0) / 1e6);
    return 0;
}
