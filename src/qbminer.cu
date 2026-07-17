#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <netdb.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <csignal>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

struct Options {
  std::string host = "mine.prismpool.io";
  int port = 4335;
  std::string user = "qb1zhqwu3s35yyrfsqlr42snrzx7xwgdqhx89vdaupdc4nuyt95y8v4qxttk86.4090vps";
  std::string pass = "x";
  int device = -1;
  int blocks = 131072;
  int threads = 256;
  bool dashboard = true;
  std::string log_file = "qbminer.log";
  double share_factor = 1.0;
  bool debug_shares = false;
};

static std::mutex log_mutex;
static std::ofstream event_log;

struct GpuStats {
  std::string name;
  std::string worker;
  std::string status = "starting";
  std::string last_job = "-";
  std::string last_event = "-";
  double difficulty = 0.0;
  double mhps = 0.0;
  uint64_t total_hashes = 0;
  uint64_t submitted = 0;
  uint64_t accepted = 0;
  uint64_t rejected = 0;
  uint64_t stale = 0;
};

static std::vector<GpuStats> gpu_stats;
static std::string dashboard_pool;
static std::string dashboard_user;
static std::chrono::steady_clock::time_point dashboard_started;
static std::atomic<bool> dashboard_done{false};
static bool dashboard_interactive = true;

static void restore_terminal() {
  if (dashboard_interactive) {
    std::cout << "\033[?25h\033[?1049l" << std::flush;
  }
}

static void handle_signal(int sig) {
  restore_terminal();
  std::_Exit(128 + sig);
}

static std::string timestamp() {
  auto now = std::chrono::system_clock::now();
  std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
  localtime_r(&t, &tm);
  char buf[32];
  strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
  return buf;
}

struct Job {
  std::string id, prevhash, coinb1, coinb2, version, nbits, ntime;
  std::vector<std::string> branches;
  bool clean = false;
};

struct GpuResult {
  uint32_t found;
  uint32_t nonce;
};

struct WorkData {
  uint32_t midstate[8];
  uint32_t tail[3];
};

__host__ __device__ static inline uint32_t rotr32(uint32_t x, uint32_t n) {
  return (x >> n) | (x << (32 - n));
}

__host__ __device__ static inline uint32_t bswap32(uint32_t x) {
  return ((x & 0x000000ffU) << 24) | ((x & 0x0000ff00U) << 8) |
         ((x & 0x00ff0000U) >> 8) | ((x & 0xff000000U) >> 24);
}

__host__ __device__ static void sha256_compress(uint32_t s[8], const uint8_t block[64]) {
  const uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
  };
  uint32_t w[64];
  for (int i = 0; i < 16; i++) {
    w[i] = ((uint32_t)block[i*4] << 24) | ((uint32_t)block[i*4+1] << 16) |
           ((uint32_t)block[i*4+2] << 8) | block[i*4+3];
  }
  for (int i = 16; i < 64; i++) {
    uint32_t s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3);
    uint32_t s1 = rotr32(w[i-2], 17) ^ rotr32(w[i-2], 19) ^ (w[i-2] >> 10);
    w[i] = w[i-16] + s0 + w[i-7] + s1;
  }

  uint32_t a=s[0], b=s[1], c=s[2], d=s[3], e=s[4], f=s[5], g=s[6], h=s[7];
  for (int i = 0; i < 64; i++) {
    uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
    uint32_t ch = (e & f) ^ ((~e) & g);
    uint32_t t1 = h + S1 + ch + K256[i] + w[i];
    uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
    uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    uint32_t t2 = S0 + maj;
    h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
  }
  s[0]+=a; s[1]+=b; s[2]+=c; s[3]+=d; s[4]+=e; s[5]+=f; s[6]+=g; s[7]+=h;
}

__host__ __device__ static void sha256_compress_words(uint32_t s[8], uint32_t w0[16]) {
  const uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
  };
  uint32_t w[64];
  #pragma unroll
  for (int i = 0; i < 16; i++) w[i] = w0[i];
  #pragma unroll
  for (int i = 16; i < 64; i++) {
    uint32_t s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3);
    uint32_t s1 = rotr32(w[i-2], 17) ^ rotr32(w[i-2], 19) ^ (w[i-2] >> 10);
    w[i] = w[i-16] + s0 + w[i-7] + s1;
  }

  uint32_t a=s[0], b=s[1], c=s[2], d=s[3], e=s[4], f=s[5], g=s[6], h=s[7];
  #pragma unroll
  for (int i = 0; i < 64; i++) {
    uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
    uint32_t ch = (e & f) ^ ((~e) & g);
    uint32_t t1 = h + S1 + ch + K256[i] + w[i];
    uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
    uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    uint32_t t2 = S0 + maj;
    h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
  }
  s[0]+=a; s[1]+=b; s[2]+=c; s[3]+=d; s[4]+=e; s[5]+=f; s[6]+=g; s[7]+=h;
}

__host__ __device__ static void sha256_bytes(const uint8_t *data, size_t len, uint8_t out[32]) {
  uint32_t s[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  uint8_t block[64];
  size_t off = 0;
  while (len - off >= 64) {
    sha256_compress(s, data + off);
    off += 64;
  }
  size_t rem = len - off;
  for (size_t i = 0; i < rem; i++) block[i] = data[off+i];
  block[rem++] = 0x80;
  if (rem > 56) {
    while (rem < 64) block[rem++] = 0;
    sha256_compress(s, block);
    rem = 0;
  }
  while (rem < 56) block[rem++] = 0;
  uint64_t bits = (uint64_t)len * 8;
  for (int i = 7; i >= 0; i--) block[rem++] = (uint8_t)(bits >> (i * 8));
  sha256_compress(s, block);
  for (int i = 0; i < 8; i++) {
    out[i*4] = (uint8_t)(s[i] >> 24);
    out[i*4+1] = (uint8_t)(s[i] >> 16);
    out[i*4+2] = (uint8_t)(s[i] >> 8);
    out[i*4+3] = (uint8_t)s[i];
  }
}

static void dsha256(const std::vector<uint8_t> &in, uint8_t out[32]) {
  uint8_t tmp[32];
  sha256_bytes(in.data(), in.size(), tmp);
  sha256_bytes(tmp, 32, out);
}

__device__ static bool hash_meets_target(const uint8_t h[32], const uint8_t target[32]) {
  for (int i = 0; i < 32; i++) {
    if (h[i] < target[i]) return true;
    if (h[i] > target[i]) return false;
  }
  return true;
}

__device__ static uint8_t word_be_byte(uint32_t w, int byte_index) {
  return (uint8_t)(w >> ((3 - byte_index) * 8));
}

__device__ static bool hash_words_meet_target(const uint32_t h[8], const uint32_t target[8]) {
  // Bitcoin-style displayed hash is the byte-reversal of the SHA256d digest.
  // Stratum difficulty target is compared against that displayed big-endian value.
  #pragma unroll
  for (int i = 0; i < 32; i++) {
    int digest_index = 31 - i;
    uint8_t hb = word_be_byte(h[digest_index / 4], digest_index % 4);
    uint8_t tb = word_be_byte(target[i / 4], i % 4);
    if (hb < tb) return true;
    if (hb > tb) return false;
  }
  return true;
}

__global__ void mine_kernel(const WorkData *work, const uint32_t *target, uint32_t start_nonce, GpuResult *res) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t nonce = start_nonce + idx;
  if (res->found) return;

  uint32_t s1[8];
  #pragma unroll
  for (int i = 0; i < 8; i++) s1[i] = work->midstate[i];

  uint32_t block2[16];
  block2[0] = work->tail[0];
  block2[1] = work->tail[1];
  block2[2] = work->tail[2];
  block2[3] = bswap32(nonce);
  block2[4] = 0x80000000U;
  #pragma unroll
  for (int i = 5; i < 15; i++) block2[i] = 0;
  block2[15] = 640;
  sha256_compress_words(s1, block2);

  uint32_t s2[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  uint32_t final_block[16];
  #pragma unroll
  for (int i = 0; i < 8; i++) final_block[i] = s1[i];
  final_block[8] = 0x80000000U;
  #pragma unroll
  for (int i = 9; i < 15; i++) final_block[i] = 0;
  final_block[15] = 256;
  sha256_compress_words(s2, final_block);

  if (hash_words_meet_target(s2, target) && atomicCAS(&res->found, 0U, 1U) == 0U) {
    res->nonce = nonce;
  }
}

static std::string trim_scheme(std::string s) {
  const std::string p = "stratum+tcp://";
  if (s.rfind(p, 0) == 0) s = s.substr(p.size());
  return s;
}

static std::vector<uint8_t> hex_to_bytes(const std::string &hex) {
  std::vector<uint8_t> out;
  out.reserve(hex.size() / 2);
  for (size_t i = 0; i + 1 < hex.size(); i += 2) {
    out.push_back((uint8_t)strtoul(hex.substr(i, 2).c_str(), nullptr, 16));
  }
  return out;
}

static std::string le_hex(uint32_t v) {
  char buf[9];
  snprintf(buf, sizeof(buf), "%02x%02x%02x%02x", v & 255, (v >> 8) & 255, (v >> 16) & 255, (v >> 24) & 255);
  return buf;
}

static std::string u32_hex(uint32_t v) {
  char buf[9];
  snprintf(buf, sizeof(buf), "%08x", v);
  return buf;
}

static std::string json_get_string(const std::string &s, const std::string &key) {
  std::string pat = "\"" + key + "\":";
  size_t p = s.find(pat);
  if (p == std::string::npos) return "";
  p = s.find('"', p + pat.size());
  if (p == std::string::npos) return "";
  size_t e = s.find('"', p + 1);
  if (e == std::string::npos) return "";
  return s.substr(p + 1, e - p - 1);
}

static std::vector<std::string> quoted_strings(const std::string &s) {
  std::vector<std::string> out;
  for (size_t i = 0; i < s.size();) {
    if (s[i] != '"') { i++; continue; }
    std::string v;
    i++;
    while (i < s.size()) {
      if (s[i] == '\\' && i + 1 < s.size()) { v.push_back(s[i+1]); i += 2; continue; }
      if (s[i] == '"') { i++; break; }
      v.push_back(s[i++]);
    }
    out.push_back(v);
  }
  return out;
}

static std::string json_array_after_key(const std::string &s, const std::string &key) {
  size_t k = s.find("\"" + key + "\"");
  if (k == std::string::npos) return "";
  size_t p = s.find('[', k);
  if (p == std::string::npos) return "";

  bool in_string = false;
  bool escaped = false;
  int depth = 0;
  for (size_t i = p; i < s.size(); i++) {
    char c = s[i];
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }
    if (c == '"') {
      in_string = true;
    } else if (c == '[') {
      depth++;
    } else if (c == ']') {
      depth--;
      if (depth == 0) return s.substr(p, i - p + 1);
    }
  }
  return "";
}

static bool parse_notify(const std::string &line, Job &job) {
  std::string params = json_array_after_key(line, "params");
  auto q = quoted_strings(params);
  if (q.size() < 7) return false;

  job.id = q[0];
  job.prevhash = q[1];
  job.coinb1 = q[2];
  job.coinb2 = q[3];
  job.branches.clear();
  for (size_t i = 4; i + 3 < q.size(); i++) job.branches.push_back(q[i]);

  job.version = q[q.size() - 3];
  job.nbits = q[q.size() - 2];
  job.ntime = q[q.size() - 1];
  job.clean = line.find("true", line.find(job.ntime)) != std::string::npos;
  return job.prevhash.size() == 64 && job.version.size() == 8 &&
         job.nbits.size() == 8 && job.ntime.size() == 8;
}

static bool parse_subscribe(const std::string &line, std::string &ex1, int &ex2_size) {
  size_t r = line.find("\"result\"");
  if (r == std::string::npos) return false;

  size_t compact = line.find("[[]", r);
  if (compact != std::string::npos) {
    size_t q1 = line.find('"', compact);
    if (q1 == std::string::npos) return false;
    size_t q2 = line.find('"', q1 + 1);
    if (q2 == std::string::npos) return false;
    ex1 = line.substr(q1 + 1, q2 - q1 - 1);
    size_t comma = line.find(',', q2 + 1);
    if (comma == std::string::npos) return false;
    ex2_size = atoi(line.c_str() + comma + 1);
    return !ex1.empty() && ex2_size > 0 && ex2_size < 16;
  }

  // Handles both:
  // [[["mining.notify","..."],["mining.set_difficulty","..."]],"extranonce1",4]
  // [[], "extranonce1", 8]
  size_t result_array = line.find('[', r);
  if (result_array == std::string::npos) return false;
  size_t q1 = line.find('"', result_array);
  while (q1 != std::string::npos) {
    size_t q2_probe = line.find('"', q1 + 1);
    if (q2_probe == std::string::npos) return false;
    std::string candidate = line.substr(q1 + 1, q2_probe - q1 - 1);
    if (!candidate.empty() && candidate.find('.') == std::string::npos &&
        candidate.find("mining") == std::string::npos) {
      break;
    }
    q1 = line.find('"', q2_probe + 1);
  }
  if (q1 == std::string::npos) return false;
  size_t q2 = line.find('"', q1 + 1);
  if (q2 == std::string::npos) return false;
  ex1 = line.substr(q1 + 1, q2 - q1 - 1);

  size_t comma = line.find(',', q2 + 1);
  if (comma == std::string::npos) return false;
  ex2_size = atoi(line.c_str() + comma + 1);
  return !ex1.empty() && ex2_size > 0 && ex2_size < 16;
}

static double parse_difficulty(const std::string &line) {
  size_t p = line.find("mining.set_difficulty");
  if (p == std::string::npos) return 0.0;
  p = line.find('[', p);
  if (p == std::string::npos) return 0.0;
  return atof(line.c_str() + p + 1);
}

static void share_target_be(double diff, uint8_t out[32]) {
  for (int i = 0; i < 32; i++) out[i] = 0;
  if (diff <= 0) diff = 1.0;

  // Bitcoin/Stratum difficulty-1 target:
  // 00000000ffff0000000000000000000000000000000000000000000000000000
  unsigned __int128 words[4] = {0x00000000ffff0000ULL, 0, 0, 0};
  uint64_t d = (uint64_t)(diff + 0.5);
  if (d == 0) d = 1;

  unsigned __int128 rem = 0;
  for (int i = 0; i < 4; i++) {
    unsigned __int128 cur = (rem << 64) | words[i];
    words[i] = cur / d;
    rem = cur % d;
  }

  for (int i = 0; i < 4; i++) {
    uint64_t w = (uint64_t)words[i];
    out[i * 8 + 0] = (uint8_t)(w >> 56);
    out[i * 8 + 1] = (uint8_t)(w >> 48);
    out[i * 8 + 2] = (uint8_t)(w >> 40);
    out[i * 8 + 3] = (uint8_t)(w >> 32);
    out[i * 8 + 4] = (uint8_t)(w >> 24);
    out[i * 8 + 5] = (uint8_t)(w >> 16);
    out[i * 8 + 6] = (uint8_t)(w >> 8);
    out[i * 8 + 7] = (uint8_t)w;
  }
}

static WorkData make_work_data(const std::vector<uint8_t> &prefix76) {
  WorkData w{};
  uint32_t s[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  sha256_compress(s, prefix76.data());
  for (int i = 0; i < 8; i++) w.midstate[i] = s[i];
  for (int i = 0; i < 3; i++) {
    size_t p = 64 + i * 4;
    w.tail[i] = ((uint32_t)prefix76[p] << 24) | ((uint32_t)prefix76[p + 1] << 16) |
                ((uint32_t)prefix76[p + 2] << 8) | prefix76[p + 3];
  }
  return w;
}

static void target_words_be(const uint8_t bytes[32], uint32_t words[8]) {
  for (int i = 0; i < 8; i++) {
    words[i] = ((uint32_t)bytes[i * 4] << 24) | ((uint32_t)bytes[i * 4 + 1] << 16) |
               ((uint32_t)bytes[i * 4 + 2] << 8) | bytes[i * 4 + 3];
  }
}

static int connect_tcp(const std::string &host, int port) {
  addrinfo hints{}, *res = nullptr;
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  std::string ps = std::to_string(port);
  if (getaddrinfo(host.c_str(), ps.c_str(), &hints, &res) != 0) return -1;
  int fd = -1;
  for (addrinfo *p = res; p; p = p->ai_next) {
    fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (fd < 0) continue;
    if (connect(fd, p->ai_addr, p->ai_addrlen) == 0) break;
    close(fd);
    fd = -1;
  }
  freeaddrinfo(res);
  return fd;
}

static bool send_line(int fd, const std::string &s) {
  std::string x = s + "\n";
  return send(fd, x.data(), x.size(), 0) == (ssize_t)x.size();
}

static bool recv_line(int fd, std::string &line) {
  line.clear();
  char c;
  while (true) {
    ssize_t n = recv(fd, &c, 1, 0);
    if (n <= 0) return false;
    if (c == '\n') return true;
    if (c != '\r') line.push_back(c);
  }
}

static std::vector<uint8_t> merkle_root(const Job &job, const std::string &ex1, const std::string &ex2) {
  std::vector<uint8_t> coinbase = hex_to_bytes(job.coinb1 + ex1 + ex2 + job.coinb2);
  uint8_t root[32];
  dsha256(coinbase, root);
  std::vector<uint8_t> cur(root, root + 32);
  for (auto &bhex : job.branches) {
    std::vector<uint8_t> branch = hex_to_bytes(bhex);
    std::vector<uint8_t> both;
    both.reserve(64);
    both.insert(both.end(), cur.begin(), cur.end());
    both.insert(both.end(), branch.begin(), branch.end());
    uint8_t next[32];
    dsha256(both, next);
    cur.assign(next, next + 32);
  }
  return cur;
}

static std::vector<uint8_t> make_header76(const Job &job, const std::string &ex1, const std::string &ex2) {
  std::vector<uint8_t> h;
  auto version = hex_to_bytes(job.version);
  auto prev = hex_to_bytes(job.prevhash);
  auto merkle = merkle_root(job, ex1, ex2);
  auto ntime = hex_to_bytes(job.ntime);
  auto nbits = hex_to_bytes(job.nbits);
  h.reserve(76);
  h.insert(h.end(), version.rbegin(), version.rend());
  h.insert(h.end(), prev.rbegin(), prev.rend());
  h.insert(h.end(), merkle.rbegin(), merkle.rend());
  h.insert(h.end(), ntime.rbegin(), ntime.rend());
  h.insert(h.end(), nbits.rbegin(), nbits.rend());
  return h;
}

static void append_field(std::vector<uint8_t> &out, const std::vector<uint8_t> &field, bool reverse) {
  if (reverse) out.insert(out.end(), field.rbegin(), field.rend());
  else out.insert(out.end(), field.begin(), field.end());
}

static std::vector<uint8_t> make_header76_variant(
    const Job &job,
    const std::string &ex1,
    const std::string &ex2,
    bool rev_version,
    bool rev_prev,
    bool rev_merkle,
    bool rev_ntime,
    bool rev_nbits) {
  std::vector<uint8_t> h;
  auto version = hex_to_bytes(job.version);
  auto prev = hex_to_bytes(job.prevhash);
  auto merkle = merkle_root(job, ex1, ex2);
  auto ntime = hex_to_bytes(job.ntime);
  auto nbits = hex_to_bytes(job.nbits);
  h.reserve(76);
  append_field(h, version, rev_version);
  append_field(h, prev, rev_prev);
  append_field(h, merkle, rev_merkle);
  append_field(h, ntime, rev_ntime);
  append_field(h, nbits, rev_nbits);
  return h;
}

static Options parse_args(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "-o") {
      std::string hp = trim_scheme(next());
      size_t c = hp.rfind(':');
      if (c != std::string::npos) {
        o.host = hp.substr(0, c);
        o.port = atoi(hp.substr(c + 1).c_str());
      } else {
        o.host = hp;
      }
    } else if (a == "-u") o.user = next();
    else if (a == "-p") o.pass = next();
    else if (a == "-d") o.device = atoi(next().c_str());
    else if (a == "-b") o.blocks = atoi(next().c_str());
    else if (a == "-t") o.threads = atoi(next().c_str());
    else if (a == "--no-dashboard") o.dashboard = false;
    else if (a == "--log-file") o.log_file = next();
    else if (a == "--share-factor") o.share_factor = atof(next().c_str());
    else if (a == "--debug-shares") o.debug_shares = true;
    else if (a == "-h" || a == "--help") {
      std::cout << "qbminer -o host:port -u address.worker -p x [-d device] [-b blocks] [-t threads] [--no-dashboard] [--log-file path] [--share-factor n] [--debug-shares]\n"
                << "Default: use all CUDA GPUs. Pass -d N to mine on only GPU N.\n"
                << "Use --no-dashboard for plain one-line log output.\n"
                << "Default log file: qbminer.log\n"
                << "Default share-factor: 1.\n"
                << "Use --debug-shares to log header/hash/target for submitted shares.\n";
      exit(0);
    }
  }
  return o;
}

static std::string bytes_to_hex(const uint8_t *data, size_t len) {
  static const char *hex = "0123456789abcdef";
  std::string out;
  out.resize(len * 2);
  for (size_t i = 0; i < len; i++) {
    out[i * 2] = hex[data[i] >> 4];
    out[i * 2 + 1] = hex[data[i] & 15];
  }
  return out;
}

static std::string bytes_to_hex(const std::vector<uint8_t> &data) {
  return bytes_to_hex(data.data(), data.size());
}

static std::string reversed_hex(const uint8_t *data, size_t len) {
  std::vector<uint8_t> rev(data, data + len);
  std::reverse(rev.begin(), rev.end());
  return bytes_to_hex(rev);
}

static bool display_hash_meets_target(const uint8_t hash[32], const uint8_t target[32]) {
  for (int i = 0; i < 32; i++) {
    uint8_t hb = hash[31 - i];
    uint8_t tb = target[i];
    if (hb < tb) return true;
    if (hb > tb) return false;
  }
  return true;
}

static std::string with_gpu_worker(const std::string &user, int device, bool multi_gpu) {
  if (!multi_gpu) return user;
  return user + ".gpu" + std::to_string(device);
}

static std::string format_hashrate(double mhps) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(2);
  if (mhps >= 1000000.0) out << (mhps / 1000000.0) << " TH/s";
  else if (mhps >= 1000.0) out << (mhps / 1000.0) << " GH/s";
  else out << mhps << " MH/s";
  return out.str();
}

static std::string format_elapsed(uint64_t seconds) {
  uint64_t h = seconds / 3600;
  uint64_t m = (seconds % 3600) / 60;
  uint64_t s = seconds % 60;
  char buf[32];
  snprintf(buf, sizeof(buf), "%02llu:%02llu:%02llu",
           (unsigned long long)h, (unsigned long long)m, (unsigned long long)s);
  return buf;
}

static void log_line(int device, const std::string &msg) {
  std::lock_guard<std::mutex> lock(log_mutex);
  if (device >= 0 && device < (int)gpu_stats.size()) {
    if (msg.rfind("speed ", 0) != 0) {
      gpu_stats[device].last_event = msg;
    }
  }
  if (!dashboard_interactive) {
    std::cout << "[gpu" << device << "] " << msg << std::endl;
  }
  if (event_log.is_open()) {
    event_log << timestamp() << " [gpu" << device << "] " << msg << "\n";
    event_log.flush();
  }
}

static std::string share_reject_reason(const std::string &line) {
  if (line.find("low-difficulty") != std::string::npos || line.find("low difficulty") != std::string::npos) {
    return "low difficulty";
  }
  if (line.find("duplicate") != std::string::npos) return "duplicate";
  if (line.find("stale") != std::string::npos) return "stale";
  if (line.find("invalid") != std::string::npos) return "invalid";
  size_t p = line.find("\"error\"");
  if (p != std::string::npos) return line.substr(p, std::min<size_t>(80, line.size() - p));
  return "rejected";
}

static void dashboard_loop() {
  if (dashboard_interactive) {
    std::cout << "\033[?1049h\033[?25l" << std::flush;
  }
  while (!dashboard_done.load()) {
    std::vector<GpuStats> snap;
    {
      std::lock_guard<std::mutex> lock(log_mutex);
      snap = gpu_stats;
    }

    double total_mhps = 0.0;
    uint64_t total_hashes = 0, submitted = 0, accepted = 0, rejected = 0, stale = 0;
    for (const auto &g : snap) {
      total_mhps += g.mhps;
      total_hashes += g.total_hashes;
      submitted += g.submitted;
      accepted += g.accepted;
      rejected += g.rejected;
      stale += g.stale;
    }

    auto now = std::chrono::steady_clock::now();
    uint64_t elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - dashboard_started).count();
    double avg_mhps = elapsed > 0 ? (double)total_hashes / (double)elapsed / 1e6 : 0.0;

    std::ostringstream out;
    out << "\033[H\033[2J";
    out << "qbminer - Qbit PRISM CUDA miner\n";
    out << "Pool: " << dashboard_pool << "    Address/worker: " << dashboard_user << "\n";
    out << "Uptime: " << format_elapsed(elapsed)
        << "    Total: " << format_hashrate(total_mhps)
        << "    Average: " << format_hashrate(avg_mhps) << "\n";
    out << "Shares: accepted " << accepted
        << " | rejected " << rejected
        << " | stale " << stale
        << " | submitted " << submitted << "\n\n";

    out << std::left
        << std::setw(5) << "GPU"
        << std::setw(27) << "Device"
        << std::setw(12) << "Speed"
        << std::setw(10) << "Diff"
        << std::setw(10) << "Acc/Rej"
        << std::setw(11) << "Submitted"
        << std::setw(18) << "Job"
        << std::setw(13) << "Status"
        << "Event\n";
    out << std::string(110, '-') << "\n";

    for (size_t i = 0; i < snap.size(); i++) {
      const auto &g = snap[i];
      std::string accrej = std::to_string(g.accepted) + "/" + std::to_string(g.rejected);
      out << std::left
          << std::setw(5) << ("#" + std::to_string(i))
          << std::setw(27) << g.name.substr(0, 26)
          << std::setw(12) << format_hashrate(g.mhps)
          << std::setw(10) << (uint64_t)g.difficulty
          << std::setw(10) << accrej
          << std::setw(11) << g.submitted
          << std::setw(18) << g.last_job.substr(0, 17)
          << std::setw(13) << g.status.substr(0, 12)
          << g.last_event.substr(0, 80) << "\n";
    }

    std::cout << out.str() << std::flush;
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }
  if (dashboard_interactive) {
    std::cout << "\033[?25h\033[?1049l" << std::flush;
  }
}

static int run_device(Options opt, int device, bool multi_gpu) {
  opt.device = device;
  opt.user = with_gpu_worker(opt.user, device, multi_gpu);

  CUDA_CHECK(cudaSetDevice(opt.device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, opt.device));
  {
    std::lock_guard<std::mutex> lock(log_mutex);
    if (opt.device >= 0 && opt.device < (int)gpu_stats.size()) {
      gpu_stats[opt.device].name = prop.name;
      gpu_stats[opt.device].worker = opt.user;
      gpu_stats[opt.device].status = "connecting";
    }
  }
  log_line(opt.device, std::string("qbminer CUDA SHA256d on ") + prop.name);
  log_line(opt.device, "Connecting to " + opt.host + ":" + std::to_string(opt.port));

  int fd = connect_tcp(opt.host, opt.port);
  if (fd < 0) {
    log_line(opt.device, "connect failed");
    return 1;
  }

  send_line(fd, "{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[\"qbminer/0.1\"]}");
  std::string line, ex1;
  int ex2_size = 0;
  while (recv_line(fd, line)) {
    log_line(opt.device, "< " + line);
    if (line.find("\"id\": 1") != std::string::npos || line.find("\"id\":1") != std::string::npos) {
      if (parse_subscribe(line, ex1, ex2_size)) break;
      log_line(opt.device, "failed to parse subscribe response");
    }
  }
  if (ex1.empty()) {
    log_line(opt.device, "subscribe failed");
    return 1;
  }
  log_line(opt.device, "extranonce1=" + ex1 + " extranonce2_size=" + std::to_string(ex2_size));
  {
    std::lock_guard<std::mutex> lock(log_mutex);
    gpu_stats[opt.device].status = "authorizing";
  }

  std::ostringstream auth;
  auth << "{\"id\":2,\"method\":\"mining.authorize\",\"params\":[\"" << opt.user << "\",\"" << opt.pass << "\"]}";
  send_line(fd, auth.str());

  WorkData *d_work = nullptr;
  uint32_t *d_target = nullptr;
  GpuResult *d_res = nullptr, h_res{};
  CUDA_CHECK(cudaMalloc(&d_work, sizeof(WorkData)));
  CUDA_CHECK(cudaMalloc(&d_target, 8 * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_res, sizeof(GpuResult)));

  Job job;
  double diff = 1.0;
  uint64_t ex2_counter = 0;
  auto last_report = std::chrono::steady_clock::now();
  uint64_t hashes_since = 0;

  while (true) {
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(fd, &rfds);
    timeval tv{0, 1000};
    if (select(fd + 1, &rfds, nullptr, nullptr, &tv) > 0) {
      if (!recv_line(fd, line)) {
        log_line(opt.device, "pool disconnected");
        return 1;
      }
      if (line.find("mining.set_difficulty") != std::string::npos) {
        diff = parse_difficulty(line);
        {
          std::lock_guard<std::mutex> lock(log_mutex);
          gpu_stats[opt.device].difficulty = diff;
        }
        log_line(opt.device, "difficulty " + std::to_string(diff));
      } else if (line.find("mining.notify") != std::string::npos) {
        if (parse_notify(line, job)) {
          {
            std::lock_guard<std::mutex> lock(log_mutex);
            gpu_stats[opt.device].last_job = job.id;
            gpu_stats[opt.device].status = "mining";
            gpu_stats[opt.device].last_event = "new work";
          }
          ex2_counter = 0;
        } else {
          log_line(opt.device, "failed to parse notify: " + line);
        }
      } else if (line.find("\"id\": 4") != std::string::npos || line.find("\"id\":4") != std::string::npos) {
        std::lock_guard<std::mutex> lock(log_mutex);
        std::string verdict;
        if (line.find("\"result\": true") != std::string::npos || line.find("\"result\":true") != std::string::npos) {
          gpu_stats[opt.device].accepted++;
          verdict = "share accepted";
          gpu_stats[opt.device].last_event = verdict;
        } else if (line.find("stale") != std::string::npos) {
          gpu_stats[opt.device].stale++;
          gpu_stats[opt.device].rejected++;
          verdict = "share stale";
          gpu_stats[opt.device].last_event = verdict;
        } else {
          gpu_stats[opt.device].rejected++;
          verdict = "share rejected: " + share_reject_reason(line);
          gpu_stats[opt.device].last_event = verdict;
        }
        if (!dashboard_interactive) {
          std::cout << "[gpu" << opt.device << "] < " << line << std::endl;
        }
        if (event_log.is_open()) {
          event_log << timestamp() << " [gpu" << opt.device << "] " << verdict << "\n";
          event_log << timestamp() << " [gpu" << opt.device << "] < " << line << "\n";
          event_log.flush();
        }
      } else {
        log_line(opt.device, "< " + line);
      }
    }

    if (job.id.empty()) continue;

    std::ostringstream ex2ss;
    ex2ss << std::hex;
    for (int i = ex2_size - 1; i >= 0; i--) {
      uint8_t b = (ex2_counter >> (i * 8)) & 0xff;
      char buf[3];
      snprintf(buf, sizeof(buf), "%02x", b);
      ex2ss << buf;
    }
    std::string ex2 = ex2ss.str();
    auto prefix = make_header76(job, ex1, ex2);
    if (prefix.size() != 76) {
      log_line(opt.device, "bad header size");
      return 1;
    }
    WorkData work = make_work_data(prefix);
    CUDA_CHECK(cudaMemcpy(d_work, &work, sizeof(work), cudaMemcpyHostToDevice));

    uint8_t target_bytes[32];
    uint32_t target[8];
    share_target_be(diff * opt.share_factor, target_bytes);
    target_words_be(target_bytes, target);
    CUDA_CHECK(cudaMemcpy(d_target, target, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    uint32_t batch = (uint32_t)(opt.blocks * opt.threads);
    for (uint64_t start64 = 0; start64 < 0x100000000ULL; start64 += batch) {
      uint32_t start = (uint32_t)start64;
      h_res.found = 0; h_res.nonce = 0;
      CUDA_CHECK(cudaMemcpy(d_res, &h_res, sizeof(h_res), cudaMemcpyHostToDevice));
      mine_kernel<<<opt.blocks, opt.threads>>>(d_work, d_target, start, d_res);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpy(&h_res, d_res, sizeof(h_res), cudaMemcpyDeviceToHost));
      hashes_since += batch;
      {
        std::lock_guard<std::mutex> lock(log_mutex);
        gpu_stats[opt.device].total_hashes += batch;
      }

      auto now = std::chrono::steady_clock::now();
      double sec = std::chrono::duration<double>(now - last_report).count();
      if (sec >= 5.0) {
        double mhps = hashes_since / sec / 1e6;
        {
          std::lock_guard<std::mutex> lock(log_mutex);
          gpu_stats[opt.device].mhps = mhps;
          gpu_stats[opt.device].difficulty = diff;
          gpu_stats[opt.device].status = "mining";
        }
        std::ostringstream speed;
        speed << "speed " << mhps << " MH/s diff " << diff;
        log_line(opt.device, speed.str());
        hashes_since = 0;
        last_report = now;
      }

      if (h_res.found) {
        std::string nonce_header_hex = le_hex(h_res.nonce);
        std::string nonce_submit_hex = nonce_header_hex;
        std::vector<uint8_t> header = prefix;
        header.push_back((uint8_t)(h_res.nonce & 0xff));
        header.push_back((uint8_t)((h_res.nonce >> 8) & 0xff));
        header.push_back((uint8_t)((h_res.nonce >> 16) & 0xff));
        header.push_back((uint8_t)((h_res.nonce >> 24) & 0xff));
        uint8_t cpu_hash[32];
        dsha256(header, cpu_hash);
        std::ostringstream sub;
        sub << "{\"id\":4,\"method\":\"mining.submit\",\"params\":[\""
            << opt.user << "\",\"" << job.id << "\",\"" << ex2 << "\",\""
            << job.ntime << "\",\"" << nonce_submit_hex << "\"]}";
        log_line(opt.device, "share nonce " + nonce_submit_hex);
        {
          std::lock_guard<std::mutex> lock(log_mutex);
          gpu_stats[opt.device].submitted++;
          gpu_stats[opt.device].last_event = "share submitted " + nonce_submit_hex;
        }
        if (opt.debug_shares && event_log.is_open()) {
          event_log << timestamp() << " [gpu" << opt.device << "] DEBUG_SHARE_BEGIN\n";
          event_log << "worker=" << opt.user << "\n";
          event_log << "job=" << job.id << "\n";
          event_log << "difficulty=" << diff << "\n";
          event_log << "share_factor=" << opt.share_factor << "\n";
          event_log << "extranonce1=" << ex1 << "\n";
          event_log << "extranonce2=" << ex2 << "\n";
          event_log << "ntime=" << job.ntime << "\n";
          event_log << "nonce_header_le=" << nonce_header_hex << "\n";
          event_log << "nonce_submit=" << nonce_submit_hex << "\n";
          event_log << "nonce_submit_mode=header_le\n";
          event_log << "nonce_u32=" << h_res.nonce << "\n";
          event_log << "submit=" << sub.str() << "\n";
          event_log << "version=" << job.version << "\n";
          event_log << "prevhash=" << job.prevhash << "\n";
          event_log << "nbits=" << job.nbits << "\n";
          event_log << "coinb1=" << job.coinb1 << "\n";
          event_log << "coinb2=" << job.coinb2 << "\n";
          event_log << "merkle_branches=" << job.branches.size() << "\n";
          for (size_t bi = 0; bi < job.branches.size(); bi++) {
            event_log << "branch" << bi << "=" << job.branches[bi] << "\n";
          }
          event_log << "header_hex=" << bytes_to_hex(header) << "\n";
          event_log << "hash_raw=" << bytes_to_hex(cpu_hash, 32) << "\n";
          event_log << "hash_reversed=" << reversed_hex(cpu_hash, 32) << "\n";
          event_log << "target=" << bytes_to_hex(target_bytes, 32) << "\n";
          event_log << "variant_passes_begin\n";
          int variant_passes = 0;
          for (int rv = 0; rv <= 1; rv++) {
            for (int rp = 0; rp <= 1; rp++) {
              for (int rm = 0; rm <= 1; rm++) {
                for (int rt = 0; rt <= 1; rt++) {
                  for (int rb = 0; rb <= 1; rb++) {
                    auto hp = make_header76_variant(job, ex1, ex2, rv, rp, rm, rt, rb);
                    for (int nonce_mode = 0; nonce_mode <= 1; nonce_mode++) {
                      std::vector<uint8_t> vh = hp;
                      if (nonce_mode == 0) {
                        vh.push_back((uint8_t)(h_res.nonce & 0xff));
                        vh.push_back((uint8_t)((h_res.nonce >> 8) & 0xff));
                        vh.push_back((uint8_t)((h_res.nonce >> 16) & 0xff));
                        vh.push_back((uint8_t)((h_res.nonce >> 24) & 0xff));
                      } else {
                        vh.push_back((uint8_t)((h_res.nonce >> 24) & 0xff));
                        vh.push_back((uint8_t)((h_res.nonce >> 16) & 0xff));
                        vh.push_back((uint8_t)((h_res.nonce >> 8) & 0xff));
                        vh.push_back((uint8_t)(h_res.nonce & 0xff));
                      }
                      uint8_t vh_hash[32];
                      dsha256(vh, vh_hash);
                      if (display_hash_meets_target(vh_hash, target_bytes)) {
                        variant_passes++;
                        event_log << "variant_pass"
                                  << " rev_version=" << rv
                                  << " rev_prev=" << rp
                                  << " rev_merkle=" << rm
                                  << " rev_ntime=" << rt
                                  << " rev_nbits=" << rb
                                  << " nonce_mode=" << (nonce_mode == 0 ? "le" : "be")
                                  << " hash_reversed=" << reversed_hex(vh_hash, 32)
                                  << " header_hex=" << bytes_to_hex(vh)
                                  << "\n";
                      }
                    }
                  }
                }
              }
            }
          }
          event_log << "variant_passes_count=" << variant_passes << "\n";
          event_log << "variant_passes_end\n";
          event_log << timestamp() << " [gpu" << opt.device << "] DEBUG_SHARE_END\n";
          event_log.flush();
        }
        send_line(fd, sub.str());
      }

      FD_ZERO(&rfds);
      FD_SET(fd, &rfds);
      timeval tv2{0, 0};
      if (select(fd + 1, &rfds, nullptr, nullptr, &tv2) > 0) break;
    }
    ex2_counter++;
    {
      std::lock_guard<std::mutex> lock(log_mutex);
      if (opt.device >= 0 && opt.device < (int)gpu_stats.size()) {
        gpu_stats[opt.device].last_event = "next extranonce " + std::to_string(ex2_counter);
      }
    }
  }
}

int main(int argc, char **argv) {
  Options opt = parse_args(argc, argv);
  dashboard_interactive = opt.dashboard && isatty(STDOUT_FILENO);
  if (!opt.log_file.empty()) {
    event_log.open(opt.log_file, std::ios::app);
    if (!event_log) {
      std::cerr << "Could not open log file: " << opt.log_file << "\n";
      return 1;
    }
  }
  if (dashboard_interactive) {
    std::atexit(restore_terminal);
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);
  }
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  if (count <= 0) {
    std::cerr << "No CUDA GPUs found\n";
    return 1;
  }

  if (opt.device >= 0) {
    if (opt.device >= count) {
      std::cerr << "Requested GPU " << opt.device << " but only " << count << " CUDA GPU(s) found\n";
      return 1;
    }
    dashboard_pool = opt.host + ":" + std::to_string(opt.port);
    dashboard_user = opt.user;
    dashboard_started = std::chrono::steady_clock::now();
    gpu_stats.assign(count, GpuStats{});
    std::thread dashboard;
    if (dashboard_interactive) dashboard = std::thread(dashboard_loop);
    int rc = run_device(opt, opt.device, false);
    dashboard_done = true;
    if (dashboard.joinable()) dashboard.join();
    return rc;
  }

  dashboard_pool = opt.host + ":" + std::to_string(opt.port);
  dashboard_user = opt.user;
  dashboard_started = std::chrono::steady_clock::now();
  gpu_stats.assign(count, GpuStats{});
  std::thread dashboard;
  if (dashboard_interactive) dashboard = std::thread(dashboard_loop);
  std::vector<std::thread> threads;
  std::atomic<int> failures{0};
  for (int d = 0; d < count; d++) {
    threads.emplace_back([&, d]() {
      int rc = run_device(opt, d, count > 1);
      if (rc != 0) failures++;
    });
  }
  for (auto &t : threads) t.join();
  dashboard_done = true;
  if (dashboard.joinable()) dashboard.join();
  return failures.load() == 0 ? 0 : 1;
}
