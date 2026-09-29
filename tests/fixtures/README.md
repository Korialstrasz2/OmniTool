# Generated audio fixtures

`generated-silence.zip` contains five 0.2-second mono silent audio files, created locally using FFmpeg's `anullsrc` input and the libmp3lame, FLAC, AAC, libvorbis, and libopus encoders. They contain only the synthetic metadata Fixture Artist / Fixture Track / Fixture Album. No real recording, lyrics, user files, credentials, or network download is included.

Tests read named entries with `ZipFile.read`; they never extract caller-provided archives. These fixtures allow genuine Mutagen tag read/write/read-back tests on every CI platform without requiring an FFmpeg executable at test time. Test lyric text is original synthetic fixture text.
