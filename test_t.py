from app.extraction.youtube_fetch import fetch_transcript
result = fetch_transcript('https://www.youtube.com/watch?v=aircAruvnKk')
print(result[:500])