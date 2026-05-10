import acoustid
import chromaprint
from pydub import AudioSegment

import json
import concurrent
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm

class FingerprintIndex:
    """
    Build audio fingerprint database.
    """
    def __init__(self):
        self.title: str = None
        self.chunks: list[dict] = []
        self.labels: dict = {}

    def build(self, audio_path:str, interval_ms:int = 30_000, workers:int = 4) -> None:
        """
        Load audio file, split into overlapping chunks, and fingerprint
        each chunk in parallel.
        """
        print(f"[index] Loading audio: {audio_path}")
        audio_path = Path(audio_path)
        self.title = audio_path.stem
        audio = AudioSegment.from_file(audio_path)

        duration_ms = len(audio)
        print(f"[index] Audio duration: {duration_ms / 1000:.1f}s")

        jobs = []
        for start in range(0, duration_ms, int(interval_ms/2)):
            end = min(start + interval_ms, duration_ms)
            jobs.append({
                'audio': audio[start:end],
                'start': start,
                'end': end
            })
        
        print(f"[index] Processing {len(jobs)} chunks (parallel, {workers} workers)")
        chunks = run_jobs(create_chunk, jobs, workers)
        chunks = [c for c in chunks if c is not None]

        self.chunks = chunks
        self.labels = {c['name']: 0 for c in chunks}
        print(f"[index] Completed with {len(self.chunks)} non-silent chunks")

    def split_chunks(self, audio_path:str, chunks_dir:str, ext:str = "flac", frame_rate:int = 16_000, channels: int = 1):
        """
        Load audio, convert to 16khz mono, split based on chunks
        """
        chunks_dir = Path(chunks_dir)
        chunks_dir.mkdir(exist_ok=True)

        print(f"[index] Encoding audio to sr={frame_rate}")
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(frame_rate)
        audio = audio.set_channels(channels)

        jobs = []
        for chunk in self.chunks:
            name, start, end, fp = chunk.values()
            jobs.append({
                'audio': audio[start:end],
                'path': chunks_dir / f"{self.title}_{name}.{ext}",
                'ext': ext
            })
            
        print(f"[index] Processing {len(jobs)} chunks")
        chunks = run_jobs(export_audio, jobs, 4)
    
    def find_match(self, audio_path:str, score:int = 1, min_similarity:float = 0.6):
        audio = AudioSegment.from_file(audio_path)
        clip_fp = get_fingerprint(audio)

        for chunk in self.chunks:
            chunk_name = chunk['name']
            chunk_fp = chunk['fp']
            matches = match_fingerprints(chunk_fp, clip_fp)
            if matches:
                similarity, offset = matches[0]
                if similarity > min_similarity:
                    self.labels[chunk_name] = self.labels.get(chunk_name, 0) + score

    def save(self, path:str) -> None:
        Path(path).parent.mkdir(exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "title": self.title,
                "chunks": self.chunks,
                "labels": self.labels
            }, f)
        print(f"[index] Saved to {path}")

    def load(self, path:str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.title = data['title']
        self.chunks = data['chunks']
        self.labels = data['labels']
        print(f"[index] Loaded {len(self.chunks)} chunks from {path}")

def find_files(folder_path: str, extensions: list[str]) -> list:
    input_path = Path(folder_path)
    file_paths = [
        file for file in input_path.iterdir()
        if file.is_file() and file.suffix.lower() in extensions
    ]
    return file_paths

def run_jobs(fun, jobs, workers):
    results: list[dict] = [None] * len(jobs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(fun, **job): i
            for i, job in enumerate(jobs)
        }
        for future in tqdm(concurrent.futures.as_completed(future_map), total=len(jobs), desc="Processing"):
            i = future_map[future]
            try:
                results[i] = future.result()
            except Exception as e:
                print(f"  [warn] chunk {i} failed: {e}")
    return results

# Build
def is_silent(segment: AudioSegment, threshold_dbfs: float = -60) -> bool:
    """Return True if the segment's average energy is below threshold."""
    return segment.dBFS < threshold_dbfs

def audio_to_raw_buffer(segment: AudioSegment):
    """Convert pydub AudioSegment to raw PCM buffer suitable for acoustid."""
    raw = segment.raw_data
    arr = np.frombuffer(raw, dtype=np.int16)
    return arr.tobytes()

def ms_to_hms(ms: int) -> str:
    """Format miliseconds as HH:MM:SS.mmm string."""
    hours, remainder = divmod(ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}{minutes:02d}{seconds:02d}"

def get_fingerprint(audio: AudioSegment) -> list:
    raw = audio_to_raw_buffer(audio)
    fp_encoded = acoustid.fingerprint(audio.frame_rate, audio.channels, iter([raw]))
    fp_array, _ = chromaprint.decode_fingerprint(fp_encoded)
    return fp_array

def create_chunk(audio: AudioSegment, start: int, end: int) -> dict:
    if is_silent(audio):
        print(f"  [warn] skipping silent chunk {start} → {end}")
        return None
        
    return {
        'name': ms_to_hms(start),
        'start': start,
        'end': end,
        'fp': get_fingerprint(audio)
    }

# Find
def invert_index(fp_list: list[int]) -> dict[int, list[int]]:
    """Build inverted index: value → [positions]."""
    inv = {}
    for i, v in enumerate(fp_list):
        inv[v] = i
    return inv

def bit_error_rate(fp_a: list[int], fp_b: list[int], offset: int) -> float:
    """
    Compute the bit error rate between two fingerprint arrays aligned at `offset`.
    Lower BER = better match.  Returns a value in [0, 1].
    """
    
    if offset >= 0:
        a_slice = fp_a[offset: offset + len(fp_b)]
        b_slice = fp_b[: len(a_slice)]
    else:
        b_slice = fp_b[-offset: -offset + len(fp_a)]
        a_slice = fp_a[: len(b_slice)]
 
    if not a_slice or not b_slice:
        return 1.0
 
    length = min(len(a_slice), len(b_slice))
    errors = sum(bin(a_slice[i] ^ b_slice[i]).count("1") for i in range(length))
    return errors / (length * 32)  # 32 bits per fingerprint item

def match_fingerprints(ref_fp: list[int], query_fp: list[int]) -> list[tuple[float, int]]:
    """
    Find the best alignment offset of query_fp within ref_fp.
 
    Returns list of (ber_score, offset) sorted best-first (lowest BER first).
    BER is inverted to a similarity score: similarity = 1 - BER.
    """

    mask = (1 << 20) - 1
    ref_20   = [x & mask for x in ref_fp]
    query_20 = [x & mask for x in query_fp]

    common = set(ref_20) & set(query_20)
    if not common: return []
    
    ref_inv   = invert_index(ref_20)
    query_inv = invert_index(query_20)

    offset_counts = {}
    for val in common:
        ref_positions   = ref_inv[val]
        query_positions = query_inv[val]
        o = ref_positions - query_positions  # This offset represents potential alignment points
        offset_counts[o] = offset_counts.get(o, 0) + 1
    
    matches = []
    for offset, counts in offset_counts.items():
        if counts < 3: continue
        ber = bit_error_rate(ref_fp, query_fp, offset)
        similarity = 1.0 - ber
        matches.append((similarity, offset))
    matches.sort(reverse=True)
    return matches

# Export
def export_audio(audio, path, ext):
    audio.export(path, format = ext)