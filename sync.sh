#!/bin/bash

# 定义远程仓库的名称
remote_name="origin"

# 定义要同步的分支名称
branch_name="main"

# 添加所有修改
git add .

# 提交修改
git commit -m "自动同步"

# 从远程仓库拉取最新的代码
git fetch $remote_name $branch_name

# 检查是否存在冲突
if git diff --name-only $remote_name/$branch_name; then
  echo "存在冲突，正在尝试自动合并..."
  # 尝试自动合并
  if ! git merge $remote_name/$branch_name; then
    echo "自动合并失败，请手动解决冲突。"
    exit 1
  fi
fi

# 将本地的修改推送到远程仓库
git push $remote_name $branch_name

echo "同步完成。"