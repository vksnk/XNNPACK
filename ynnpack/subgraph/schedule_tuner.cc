// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/subgraph/schedule_tuner.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "ynnpack/base/log.h"

namespace ynn {

namespace {

enum class tuner_mode { record, replay, random };

uint64_t fnv1a(const char* s) {
  uint64_t h = 0xcbf29ce484222325ULL;
  for (; *s; ++s) {
    h = (h ^ static_cast<uint8_t>(*s)) * 0x100000001b3ULL;
  }
  return h;
}

uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

struct recorded_decision {
  std::string key;
  std::string kind;
  int num_options;
  int choice;
};

// The per-pipeline decision sequence currently being made. Thread-local so
// concurrent runtime builds don't interleave their sequences.
struct pipeline_sequence {
  uint64_t key = 0;
  bool active = false;
  std::vector<recorded_decision> decisions;
  // The replayed choices for this pipeline, or nullptr.
  const std::map<std::string, int>* replay = nullptr;
};

thread_local pipeline_sequence current_sequence;

struct tuner_state {
  tuner_mode mode;
  std::string path;
  uint64_t seed = 0;

  std::mutex mutex;
  // Replay: choices per pipeline key per decision key.
  std::map<uint64_t, std::map<std::string, int>> replay_choices;
  // Record: pipelines already written, and the output stream.
  std::set<uint64_t> recorded_pipelines;
  std::ofstream record_file;
};

tuner_state* state = nullptr;

void load_replay_file(tuner_state& s) {
  std::ifstream in(s.path);
  if (!in) {
    YNN_LOG_ERROR() << "YNN_SCHED_TUNE: cannot read replay file " << s.path;
    return;
  }
  std::map<std::string, int>* pipeline = nullptr;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream tokens(line);
    std::string first;
    tokens >> first;
    if (first == "pipeline") {
      uint64_t key = 0;
      tokens >> std::hex >> key;
      pipeline = &s.replay_choices[key];
    } else if (pipeline) {
      std::string kind;
      int num_options = 0;
      int choice = 0;
      if (tokens >> kind >> num_options >> choice) {
        (*pipeline)[first] = choice;
      }
    }
  }
}

tuner_state* make_state() {
  const char* env = getenv("YNN_SCHED_TUNE");
  if (!env) return nullptr;
  std::string value = env;
  auto s = std::make_unique<tuner_state>();
  if (value.rfind("record:", 0) == 0) {
    s->mode = tuner_mode::record;
    s->path = value.substr(7);
  } else if (value.rfind("replay:", 0) == 0) {
    s->mode = tuner_mode::replay;
    s->path = value.substr(7);
    load_replay_file(*s);
  } else if (value.rfind("random:", 0) == 0) {
    s->mode = tuner_mode::random;
    s->seed = strtoull(value.substr(7).c_str(), nullptr, 0);
  } else {
    YNN_LOG_ERROR() << "YNN_SCHED_TUNE: unknown mode '" << value
                    << "', expected record:<path>, replay:<path> or "
                       "random:<seed>";
    return nullptr;
  }
  return s.release();
}

}  // namespace

schedule_tuner* schedule_tuner::get() {
  static std::once_flag once;
  static schedule_tuner* instance = nullptr;
  std::call_once(once, [] {
    state = make_state();
    if (state) {
      static schedule_tuner tuner;
      instance = &tuner;
    }
  });
  return instance;
}

void schedule_tuner::begin_pipeline(uint64_t key) {
  pipeline_sequence& seq = current_sequence;
  seq.key = key;
  seq.active = true;
  seq.decisions.clear();
  seq.replay = nullptr;
  if (state->mode == tuner_mode::replay) {
    std::lock_guard<std::mutex> lock(state->mutex);
    auto it = state->replay_choices.find(key);
    if (it != state->replay_choices.end()) {
      seq.replay = &it->second;
    }
  }
}

void schedule_tuner::end_pipeline() {
  pipeline_sequence& seq = current_sequence;
  if (state->mode == tuner_mode::record && seq.active) {
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->recorded_pipelines.insert(seq.key).second) {
      std::ofstream& out = state->record_file;
      if (!out.is_open()) {
        out.open(state->path);
      }
      out << "pipeline " << std::hex << seq.key << std::dec << "\n";
      for (const recorded_decision& d : seq.decisions) {
        out << d.key << " " << d.kind << " " << d.num_options << " "
            << d.choice << "\n";
      }
      out.flush();
    }
  }
  seq.active = false;
  seq.decisions.clear();
  seq.replay = nullptr;
}

int schedule_tuner::choose(const std::string& decision_key, const char* kind,
                           int num_options, int default_choice) {
  if (num_options <= 1) return default_choice;
  pipeline_sequence& seq = current_sequence;
  if (!seq.active) return default_choice;

  int choice = default_choice;
  switch (state->mode) {
    case tuner_mode::record:
      break;
    case tuner_mode::replay:
      if (seq.replay) {
        auto it = seq.replay->find(decision_key);
        if (it != seq.replay->end()) {
          choice = std::clamp(it->second, 0, num_options - 1);
        }
      }
      break;
    case tuner_mode::random:
      choice = static_cast<int>(
          splitmix64(state->seed ^ seq.key ^ fnv1a(decision_key.c_str())) %
          num_options);
      break;
  }
  if (state->mode == tuner_mode::record) {
    seq.decisions.push_back({decision_key, kind, num_options, choice});
  }
  return choice;
}

}  // namespace ynn
