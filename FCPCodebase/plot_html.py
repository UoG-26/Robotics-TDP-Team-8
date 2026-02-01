import os
import pandas as pd
import plotly.express as px


def main():
    base_dir = os.getcwd()
    csv_path = os.path.join(base_dir, "viz_logs", "three_vs_three_run.csv")
    
    if not os.path.exists(csv_path):
        print("找不到 CSV：", csv_path)
        print("请先运行 three_vs_three.py 生成一次数据再可视化。")
        return

    df = pd.read_csv(csv_path)

    # ---------- 球轨迹 ----------
    fig_ball = px.line(
        df,
        x="ball_x",
        y="ball_y",
        title="Ball trajectory",
        labels={"ball_x": "X (m)", "ball_y": "Y (m)"},
    )

    # ---------- 取最后一个 step 的六名球员 ----------
    last_step = df["step"].max()
    df_last = df[df["step"] == last_step].copy()

    players_x = []
    players_y = []
    teams = []
    ids = []

    for i in range(6):
        x_col = f"p{i+1}_x"
        y_col = f"p{i+1}_y"

        players_x.append(df_last[x_col].values[0])
        players_y.append(df_last[y_col].values[0])
        teams.append("FCPortugal" if i < 3 else "Opponent")
        ids.append(f"p{i+1}")

    df_players = pd.DataFrame({
        "x": players_x,
        "y": players_y,
        "team": teams,
        "id": ids,
    })

    fig_players = px.scatter(
        df_players,
        x="x",
        y="y",
        color="team",
        text="id",
        title=f"Players snapshot at step {last_step}",
        labels={"x": "X (m)", "y": "Y (m)"},
    )
    fig_players.update_traces(textposition="top center")

    # ---------- 保存 HTML ----------
    out_dir = os.path.join(base_dir, "viz_logs")
    os.makedirs(out_dir, exist_ok=True)

    ball_html = os.path.join(out_dir, "ball_trajectory.html")
    players_html = os.path.join(out_dir, "players_snapshot.html")

    fig_ball.write_html(ball_html)
    fig_players.write_html(players_html)

    print("\n已生成：")
    print("  ", ball_html)
    print("  ", players_html)
    print("请用浏览器打开它们即可实时查看！")


if __name__ == "__main__":
    main()
