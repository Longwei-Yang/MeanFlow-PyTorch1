#!/bin/bash

# 用法: ./create_exp_branch.sh exp005_new_optimizer "trying new optimizer" [--push]
# 参数说明:
# $1: 分支名（如 exp005_new_optimizer）
# $2: 提交信息
# --push: 可选参数，如果加上则推送到远程

if [ $# -lt 2 ]; then
  echo "用法: $0 <branch_name> <commit_message> [--push]"
  exit 1
fi

BRANCH_NAME=$1
COMMIT_MSG=$2
PUSH_FLAG=$3

# 1. 切换回主分支并拉取更新
git checkout main && git pull origin main

# 2. 创建并切换到新分支
git checkout -b $BRANCH_NAME

# 3. 添加所有当前变更并提交
git add .
git commit -m "$BRANCH_NAME: $COMMIT_MSG"

# 4. 如果指定了 --push，则推送分支到远程
if [ "$PUSH_FLAG" == "--push" ]; then
  git push -u origin $BRANCH_NAME
fi

# # 5. 创建实验对应的 log 和 output 文件夹（可自定义）
# mkdir -p logs/$BRANCH_NAME
# mkdir -p outputs/$BRANCH_NAME

echo "✅ 分支 $BRANCH_NAME 创建并初始化成功！日志输出目录已就绪。"