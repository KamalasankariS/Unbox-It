# VHS script (https://github.com/charmbracelet/vhs) that renders docs/demo.gif.
# It calls the live Render API, so the GIF shows the real deployed service responding.
# Regenerate it with the "Demo GIF" GitHub Action (Actions tab, Run workflow) or,
# if you have VHS installed locally, with:  vhs demo.tape

Output docs/demo.gif

Set FontSize 19
Set Width 1100
Set Height 640
Set Padding 24
Set Theme "Dracula"
Set TypingSpeed 45ms

Type "# Unbox It: a live email-targeting recommender on the MIND dataset"
Enter
Sleep 1500ms

Type "# 1. Is it up?"
Enter
Type "curl -s https://unbox-it.onrender.com/health | python3 -m json.tool"
Enter
Sleep 3s

Type "# 2. Recommend articles for a reader"
Enter
Type "curl -s 'https://unbox-it.onrender.com/recommend?user_id=U000001&k=3' | python3 -m json.tool"
Enter
Sleep 4s

Type "# 3. Ask WHY it recommended that article"
Enter
Type "curl -s 'https://unbox-it.onrender.com/why?user_id=U000001&news_id=N000027' | python3 -m json.tool"
Enter
Sleep 6s
