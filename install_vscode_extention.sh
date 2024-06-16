# VS Code 追加パッケージ記述
pkglist=(ms-python.python # VSCode Python基本拡張機能
		 ms-python.vscode-pylance # VSCode Python基本拡張機能
		 shardulm94.trailing-spaces # コードの中にある余計なスペースを表示する
		 donjayamanne.python-extension-pack # VSCode Python追加拡張機能
		 GitHub.copilot  # Copilot拡張
		 GitHub.copilot-chat # Copilot拡張
		 mosapride.zenkaku # VSCodeで全角スペースを可視化する
		 kevinrose.vsc-python-indent # インデント自動調整
		 gruntfuggly.todo-tree # TO-DO記述済み箇所に飛ぶ
		 aaron-bond.better-comments # コメントをデコレーションしてくれる
		 njpwerner.autodocstring # docstring テンプレートを生成
		)

# for文で追加開始
for var in ${pkglist[@]}
do
    code --install-extension $var
done