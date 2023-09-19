import streamlit as st

# タイトルを表示
st.title("めざせ算額アプリ")

# 図形問題の画像を表示
# あなたの画像のパスに合わせて、'your_image_path.jpg'を変更してください
image_path = 'sangaku.png'
st.image(image_path, caption='図形問題の画像', use_column_width=True)

# 問題の説明を表示
st.write("問題説明をこちらに書いてください")

# 回答入力欄
answer = st.text_input("あなたの回答を入力してください:")

# 回答ボタン
if st.button('回答'):
    if answer == "正解の答え":  # 正解の答えを実際の答えに置き換えてください
        st.success("正解です！")
    else:
        st.error("間違い。もう一度トライしてみてください。")

