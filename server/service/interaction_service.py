import pandas as pd

from dtos.response_dto import ResponseDto

class InteractionService:
    def __init__(self):
        self.interactions = pd.read_csv("./data/interactions.csv")
        self.users = pd.read_csv("./data/users.csv")

    async def survey_result(self, user_id: int, movie_id_list: list) -> ResponseDto:
        for movie_id in movie_id_list:
            if self.interactions.loc[(self.interactions['user_id'] == user_id) & (self.interactions['movie_id'] == movie_id)].empty:
                self.interactions = pd.concat([self.interactions, pd.DataFrame([[user_id, movie_id]], columns=["user_id", "movie_id"])])
            
            if self.users.loc[(self.users['user_id'] == user_id) & (self.users['movie_id'] == movie_id)].empty:
                self.users = pd.concat([self.users, pd.DataFrame([[user_id, movie_id]], columns=["user_id", "movie_id"])])

        self.interactions.to_csv("./data/interactions.csv", index=False)
        self.users.to_csv("./data/users.csv", index=False)

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

    async def get_interactions(self, user_id: int) -> ResponseDto:
        user_interactions = self.interactions.loc[self.interactions['user_id'] == user_id]["movie_id"].values
        user_survey_results = self.users.loc[self.users['user_id'] == user_id]["movie_id"].values

        # 설문 결과 제외
        user_interactions = list(set(user_interactions) - set(user_survey_results))

        # Convert numpy.int64 to int
        user_interactions = [int(interaction) for interaction in user_interactions]
        
        return ResponseDto(
                status=200,
                message="Get Interactions Successfully",
                data=user_interactions
            )