import pandas as pd

from dtos.response_dto import ResponseDto

class InteractionService:
    def __init__(self):
        self.interactions = pd.read_csv("./data/interactions.csv")

    async def survey_result(self, user_id: int, movie_id_list: list) -> ResponseDto:
        for movie_id in movie_id_list:
            if self.interactions.loc[(self.interactions['user_id'] == user_id) & (self.interactions['movie_id'] == movie_id)].empty:
                self.interactions = pd.concat([self.interactions, pd.DataFrame([[user_id, movie_id]], columns=["user_id", "movie_id"])])

        self.interactions.to_csv("./data/interactions.csv", index=False)

        return ResponseDto(
                status=201,
                message="Add Survey Result Successfully"
            )

    async def control_like(self, user_id: int, movie_id: int) -> ResponseDto:
        if self.interactions.loc[(self.interactions['user_id'] == user_id) & (self.interactions['movie_id'] == movie_id)].empty:
            self.interactions = pd.concat([self.interactions, pd.DataFrame([[user_id, movie_id]], columns=["user_id", "movie_id"])])
        else:
            self.interactions = self.interactions.loc[~((self.interactions['user_id'] == user_id) & (self.interactions['movie_id'] == movie_id))]

        self.interactions.to_csv("./data/interactions.csv", index=False)

        return ResponseDto(
                status=201,
                message="Control Like Successfully"
           )
