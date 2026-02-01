import os
import pandas as pd

try:
    import plotly.express as px
except ImportError:
    raise SystemExit(
        "缺少 plotly 库，请先在虚拟环境里安装：\n"
        "  source ~/.venvs/fcp/bin/activate\n"
        "  pip install plotly\n"
    )


def build_long_df(csv_path: str) -> pd.DataFrame:
    """把 wide 格式 (p1_x, p1_y, ..., p6_x, p6_y) 展开成长表，方便做动画。"""
    df = pd.read_csv(csv_path)

    rows = []

    for _, row in df.iterrows():
        step = int(row["step"])
        t_ms = float(row["t_ms"])

        # 球
        rows.append(
            dict(
                step=step,
                t_ms=t_ms,
                id="ball",
                role="ball",
                team="Ball",
                x=row["ball_x"],
                y=row["ball_y"],
                z=row.get("ball_z", 0.0),
            )
        )

        # 6 个机器人：前 3 个视为 FCPortugal，后 3 个视为 Opponent
        for i in range(1, 7):
            x_col = f"p{i}_x"
            y_col = f"p{i}_y"

            if x_col not in row or y_col not in row:
                continue  # 容错：万一以后列名有变化

            team = "FCPortugal" if i <= 3 else "Opponent"

            rows.append(
                dict(
                    step=step,
                    t_ms=t_ms,
                    id=f"p{i}",
                    role="player",
                    team=team,
                    x=row[x_col],
                    y=row[y_col],
                    z=0.0,  # 日志里只有 2D，我们先用 2D 动画
                )
            )

    return pd.DataFrame(rows)


def main():
    base_dir = os.getcwd()
    viz_dir = os.path.join(base_dir, "viz_logs")
    csv_path = os.path.join(viz_dir, "three_vs_three_run.csv")

    if not os.path.exists(csv_path):
        print("❌ 找不到 CSV：", csv_path)
        print("请先运行 three_vs_three.py（带日志版本），再来可视化。")
        return

    print("✅ 读取数据：", csv_path)
    df_long = build_long_df(csv_path)

    # 为了让场地范围固定，不会缩放
    # Robocup 3D 标准场地：大约 x ∈ [-15, 15], y ∈ [-10, 10]，留点边缘
    x_min, x_max = -16, 16
    y_min, y_max = -11, 11

    print("✅ 构建 2D 动画图像 …")

    fig = px.scatter(
        df_long,
        x="x",
        y="y",
        animation_frame="step",       # 按 step 播放动画
        animation_group="id",         # 同一球员 / 球 在不同帧连起来
        color="team",                 # 两个队伍颜色不同
        symbol="role",                # 球和球员用不同符号
        hover_name="id",
        hover_data=["team", "t_ms"],
        title="3v3 Simulation (Top View Animation)",
        labels={"x": "X (m)", "y": "Y (m)"},
    )

    fig.update_traces(marker=dict(size=10))

    # 固定坐标轴范围，避免镜头一会儿缩一会儿放
    fig.update_xaxes(range=[x_min, x_max], zeroline=True)
    fig.update_yaxes(range=[y_min, y_max], zeroline=True, scaleanchor="x", scaleratio=1)

    # 输出到 HTML
    os.makedirs(viz_dir, exist_ok=True)
    html_path = os.path.join(viz_dir, "three_vs_three_animation.html")
    fig.write_html(html_path, auto_play=True)

    print("✅ 已生成动画 HTML：", html_path)
    print("在 Windows 里用浏览器打开它就能看到完整随时间动画啦！\n")
    print("例如在资源管理器中打开路径：")
    print(r"  \\wsl.localhost\Ubuntu\home\你的用户名\FCPCodebase\viz_logs")
    print("双击 three_vs_three_animation.html 即可。")


if __name__ == "__main__":
    main()
