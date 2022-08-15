import pandas as pd
import matplotlib.pyplot as plt
from utils.viz import create_football_field

def add_track_features(tracks, fps=59.94, snap_frame=10):
    """
    Add column features helpful for syncing with video data.
    """
    tracks = tracks.copy()
    tracks["game_play"] = (
        tracks["gameKey"].astype("str")
        + "_"
        + tracks["playID"].astype("str").str.zfill(6)
    )
    tracks["time"] = pd.to_datetime(tracks["time"])
    snap_dict = (
        tracks.query('event == "ball_snap"')
        .groupby("game_play")["time"]
        .first()
        .to_dict()
    )
    tracks["snap"] = tracks["game_play"].map(snap_dict)
    tracks["isSnap"] = tracks["snap"] == tracks["time"]
    tracks["team"] = tracks["player"].str[0].replace("H", "Home").replace("V", "Away")
    tracks["snap_offset"] = (tracks["time"] - tracks["snap"]).astype(
        "timedelta64[ms]"
    ) / 1_000
    # Estimated video frame
    tracks["est_frame"] = (
        ((tracks["snap_offset"] * fps) + snap_frame).round().astype("int")
    )
    return tracks



if __name__ == "__main__":
    root_path = "nfl-health-and-safety-helmet-assignment"
    train_player_tracking_file = "train_player_tracking.csv"
    gameKey = 57583
    player = 'H97'
    train_player_tracking_df = pd.read_csv(f"{root_path}/{train_player_tracking_file}")
    print(train_player_tracking_df.head(5))
    # let's get data only for one game to see what is it
    train_player_tracking_df_video = train_player_tracking_df.query("gameKey == @gameKey and player == @player") # 57583
    print(train_player_tracking_df_video.head())
    # number of  frames in the video
    len(train_player_tracking_df_video)
    ## add track features to sync with the video
    tr_tracking = add_track_features(train_player_tracking_df)
    game_play = "57584_000336"
    example_tracks = tr_tracking.query("game_play == @game_play and isSnap == True")
    fig, ax  = create_football_field()
    for team, d in example_tracks.groupby("team"):
        ax.scatter(d["x"], d["y"], label=team, s=65, lw=1, edgecolors="black", zorder=5)
    ax.legend().remove()
    ax.set_title(f"Tracking data for {game_play}: at snap", fontsize=15)
    plt.show()
    plt.savefig("players on field.png")
