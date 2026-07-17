#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <netdb.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
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
  int device = 0;
  int blocks = 8192;
  int threads = 256;
};

struct Job {
  std::string id, prevhash, coinb1, coinb2, version, nbits, ntime;
  std::vector<std::string> branches;
  bool clean = false;
};

struct GpuResult {
  uint32_t found;
  uint32_t nonce;
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

__global__ void mine_kernel(const uint8_t *prefix76, uint32_t start_nonce, uint32_t target_top32, GpuResult *res) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t nonce = start_nonce + idx;
  if (res->found) return;

  uint8_t header[80];
  #pragma unroll
  for (int i = 0; i < 76; i++) header[i] = prefix76[i];
  header[76] = nonce & 0xff;
  header[77] = (nonce >> 8) & 0xff;
  header[78] = (nonce >> 16) & 0xff;
  header[79] = (nonce >> 24) & 0xff;

  uint8_t h1[32], h2[32];
  sha256_bytes(header, 80, h1);
  sha256_bytes(h1, 32, h2);

  uint32_t top = ((uint32_t)h2[31] << 24) | ((uint32_t)h2[30] << 16) | ((uint32_t)h2[29] << 8) | h2[28];
  if (top <= target_top32 && atomicCAS(&res->found, 0U, 1U) == 0U) {
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

static bool parse_notify(const std::string &line, Job &job) {
  auto q = quoted_strings(line);
  if (q.size() < 9) return false;
  size_t m = 0;
  while (m < q.size() && q[m] != "mining.notify") m++;
  if (m == q.size() || m + 8 >= q.size()) return false;
  job.id = q[m+1];
  job.prevhash = q[m+2];
  job.coinb1 = q[m+3];
  job.coinb2 = q[m+4];
  job.branches.clear();
  size_t i = m + 5;
  while (i + 3 < q.size() && q[i].size() == 64) job.branches.push_back(q[i++]);
  if (i + 3 >= q.size()) return false;
  job.version = q[i++];
  job.nbits = q[i++];
  job.ntime = q[i++];
  job.clean = line.find("true", line.find(job.ntime)) != std::string::npos;
  return true;
}

static bool parse_subscribe(const std::string &line, std::string &ex1, int &ex2_size) {
  size_t r = line.find("\"result\"");
  if (r == std::string::npos) return false;

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

static uint32_t share_target_top32(double diff) {
  if (diff <= 0) diff = 1.0;
  double t = 0x0000ffffu / diff;
  if (t < 1.0) t = 1.0;
  if (t > 0xffffffffu) t = 0xffffffffu;
  return (uint32_t)t;
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
  h.insert(h.end(), prev.begin(), prev.end());
  h.insert(h.end(), merkle.begin(), merkle.end());
  h.insert(h.end(), ntime.rbegin(), ntime.rend());
  h.insert(h.end(), nbits.rbegin(), nbits.rend());
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
    else if (a == "-h" || a == "--help") {
      std::cout << "qbminer -o host:port -u address.worker -p x [-d device] [-b blocks] [-t threads]\n";
      exit(0);
    }
  }
  return o;
}

int main(int argc, char **argv) {
  Options opt = parse_args(argc, argv);
  CUDA_CHECK(cudaSetDevice(opt.device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, opt.device));
  std::cout << "qbminer CUDA SHA256d on " << prop.name << "\n";
  std::cout << "Connecting to " << opt.host << ":" << opt.port << "\n";

  int fd = connect_tcp(opt.host, opt.port);
  if (fd < 0) {
    std::cerr << "connect failed\n";
    return 1;
  }

  send_line(fd, "{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[\"qbminer/0.1\"]}");
  std::string line, ex1;
  int ex2_size = 0;
  while (recv_line(fd, line)) {
    std::cout << "< " << line << "\n";
    if (line.find("\"id\":1") != std::string::npos && parse_subscribe(line, ex1, ex2_size)) break;
  }
  if (ex1.empty()) {
    std::cerr << "subscribe failed\n";
    return 1;
  }
  std::cout << "extranonce1=" << ex1 << " extranonce2_size=" << ex2_size << "\n";

  std::ostringstream auth;
  auth << "{\"id\":2,\"method\":\"mining.authorize\",\"params\":[\"" << opt.user << "\",\"" << opt.pass << "\"]}";
  send_line(fd, auth.str());

  uint8_t *d_prefix = nullptr;
  GpuResult *d_res = nullptr, h_res{};
  CUDA_CHECK(cudaMalloc(&d_prefix, 76));
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
        std::cerr << "pool disconnected\n";
        return 1;
      }
      if (line.find("mining.set_difficulty") != std::string::npos) {
        diff = parse_difficulty(line);
        std::cout << "difficulty " << diff << "\n";
      } else if (line.find("mining.notify") != std::string::npos) {
        if (parse_notify(line, job)) {
          std::cout << "job " << job.id << " height/update received\n";
          ex2_counter = 0;
        } else {
          std::cerr << "failed to parse notify: " << line << "\n";
        }
      } else {
        std::cout << "< " << line << "\n";
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
      std::cerr << "bad header size\n";
      return 1;
    }
    CUDA_CHECK(cudaMemcpy(d_prefix, prefix.data(), 76, cudaMemcpyHostToDevice));

    uint32_t target_top = share_target_top32(diff);
    uint32_t batch = (uint32_t)(opt.blocks * opt.threads);
    for (uint32_t start = 0; start < 0xffffffffU; start += batch) {
      h_res.found = 0; h_res.nonce = 0;
      CUDA_CHECK(cudaMemcpy(d_res, &h_res, sizeof(h_res), cudaMemcpyHostToDevice));
      mine_kernel<<<opt.blocks, opt.threads>>>(d_prefix, start, target_top, d_res);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpy(&h_res, d_res, sizeof(h_res), cudaMemcpyDeviceToHost));
      hashes_since += batch;

      auto now = std::chrono::steady_clock::now();
      double sec = std::chrono::duration<double>(now - last_report).count();
      if (sec >= 5.0) {
        std::cout << "speed " << (hashes_since / sec / 1e6) << " MH/s diff " << diff << "\n";
        hashes_since = 0;
        last_report = now;
      }

      if (h_res.found) {
        std::string nonce_hex = le_hex(h_res.nonce);
        std::ostringstream sub;
        sub << "{\"id\":4,\"method\":\"mining.submit\",\"params\":[\""
            << opt.user << "\",\"" << job.id << "\",\"" << ex2 << "\",\""
            << job.ntime << "\",\"" << nonce_hex << "\"]}";
        std::cout << "share nonce " << nonce_hex << "\n";
        send_line(fd, sub.str());
      }

      FD_ZERO(&rfds);
      FD_SET(fd, &rfds);
      timeval tv2{0, 0};
      if (select(fd + 1, &rfds, nullptr, nullptr, &tv2) > 0) break;
    }
    ex2_counter++;
  }
}
