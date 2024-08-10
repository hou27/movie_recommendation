from typing import Literal
import pandas as pd

from dtos.response_dto import ResponseDto

class InteractionService:
    def __init__(self):
        # self.users = pd.read_csv("./data/users.csv")
        try:
            self.interactions = pd.read_csv("./data/interactions.csv")
            self.dislikes = pd.read_csv("./data/dislikes.csv")
        except FileNotFoundError:
            self.interactions = pd.DataFrame(columns=["user_id", "movie_id"])
            self.interactions.to_csv("./data/interactions.csv", index=False)
            self.dislikes = pd.DataFrame(columns=["user_id", "movie_id"])
            self.dislikes.to_csv("./data/dislikes.csv", index=False)

    async def survey_result(self, user_id: int, movie_id_list: list) -> ResponseDto:
        # movie_id_list = [movie_id - 1 for movie_id in movie_id_list] # db index와 data index를 맞추기 위해 1을 빼줌
        for movie_id in movie_id_list:
            if self.interactions.loc[(self.interactions['user_id'] == user_id) & (self.interactions['movie_id'] == movie_id)].empty:
                self.interactions = pd.concat([self.interactions, pd.DataFrame([[user_id, movie_id]], columns=["user_id", "movie_id"])])
            
            # if self.users.loc[(self.users['user_id'] == user_id) & (self.users['movie_id'] == movie_id)].empty:
            #     self.users = pd.concat([self.users, pd.DataFrame([[user_id, movie_id]], columns=["user_id", "movie_id"])])

        self.interactions.to_csv("./data/interactions.csv", index=False)
        # self.users.to_csv("./data/users.csv", index=False)

        return ResponseDto(
                status=201,
                message="Add Survey Result Successfully"
            )

    # async def control_like(self, user_id: int, movie_id: int) -> ResponseDto:
    #     # movie_id = movie_id - 1 # db index와 data index를 맞추기 위해 1을 빼줌
    #     if self.interactions.loc[(self.interactions['user_id'] == user_id) & (self.interactions['movie_id'] == movie_id)].empty:
    #         self.interactions = pd.concat([self.interactions, pd.DataFrame([[user_id, movie_id]], columns=["user_id", "movie_id"])])
    #     else:
    #         self.interactions = self.interactions.loc[~((self.interactions['user_id'] == user_id) & (self.interactions['movie_id'] == movie_id))]

    #     self.interactions.to_csv("./data/interactions.csv", index=False)

    #     return ResponseDto(
    #             status=201,
    #             message="Control Like Successfully"
    #        )
    
    def _toggle_interaction(self, df: pd.DataFrame, user_id: int, movie_id: int) -> tuple[pd.DataFrame, bool]:
        condition = (df['user_id'] == user_id) & (df['movie_id'] == movie_id)
        if df.loc[condition].empty:
            df = pd.concat([df, pd.DataFrame([[user_id, movie_id]], columns=["user_id", "movie_id"])])
            added = True
        else:
            df = df.loc[~condition]
            added = False
        return df, added

    def _remove_from_df(self, df: pd.DataFrame, user_id: int, movie_id: int) -> pd.DataFrame:
        return df.loc[~((df['user_id'] == user_id) & (df['movie_id'] == movie_id))]

    async def control_interaction(self, user_id: int, movie_id: int, interaction_type: Literal['like', 'dislike']) -> ResponseDto:
        if interaction_type == 'like':
            primary_df, opposite_df = self.interactions, self.dislikes
            primary_file, opposite_file = "./data/interactions.csv", "./data/dislikes.csv"
            action_name = "Like"
        else:
            primary_df, opposite_df = self.dislikes, self.interactions
            primary_file, opposite_file = "./data/dislikes.csv", "./data/interactions.csv"
            action_name = "Dislike"

        # Remove from opposite interaction
        opposite_df = self._remove_from_df(opposite_df, user_id, movie_id)

        # Toggle primary interaction
        primary_df, added = self._toggle_interaction(primary_df, user_id, movie_id)

        # Save changes
        primary_df.to_csv(primary_file, index=False)
        opposite_df.to_csv(opposite_file, index=False)

        message = f"{action_name} {'added' if added else 'removed'} successfully"
        return ResponseDto(status=201, message=message)

    async def control_like(self, user_id: int, movie_id: int) -> ResponseDto:
        return await self.control_interaction(user_id, movie_id, 'like')

    async def control_dislike(self, user_id: int, movie_id: int) -> ResponseDto:
        return await self.control_interaction(user_id, movie_id, 'dislike')

    async def get_interactions(self, user_id: int) -> ResponseDto:
        user_interactions = self.interactions.loc[self.interactions['user_id'] == user_id]["movie_id"].values
        # user_survey_results = self.users.loc[self.users['user_id'] == user_id]["movie_id"].values

        # # 설문 결과 제외
        # user_interactions = list(set(user_interactions) - set(user_survey_results))

        # Convert numpy.int64 to int
        user_interactions = [int(interaction) for interaction in user_interactions]
        
        return ResponseDto(
                status=200,
                message="Get Interactions Successfully",
                data=user_interactions
            )