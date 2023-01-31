import torch
import torch.nn as nn


class GMFModel(nn.Module):
    """The Generalized Matrix Factorization model."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int) -> None:
        """Initializes model parameters."""

        super().__init__()

        self.user_embeddings = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.item_embeddings = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim)

        # NOTE: We uniformly initialize the embeddings for equal contributions of intent
        self.user_embeddings.weight.data.uniform_(0.5, 1.0)
        self.item_embeddings.weight.data.uniform_(0.5, 1.0)

        self.affine_tranform = nn.Linear(in_features=embedding_dim, out_features=1)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass."""

        user_embeddings = self.user_embeddings(users)
        item_embeddings = self.item_embeddings(items)

        out = self.affine_tranform(user_embeddings * item_embeddings)

        return out


# Training {{{

if __name__ == "__main__":

    import pandas as pd
    import torch.optim as optim

    # DATA: wget https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
    # NOTE: The whole data pipeline can be automated (e.g., use requests)
    df = pd.read_csv("ml-latest-small/ratings.csv")
    df.drop("timestamp", axis=1, inplace=True)

    # Normalize ratings
    rating, min_rating, max_rating = df["rating"], df["rating"].min(), df["rating"].max()
    df["rating"] = (rating - min_rating) / (max_rating - min_rating)
    print(f"rating is from {df['rating'].min()} to {df['rating'].max()}")

    # Do not recommend if the rating is less than 0.5
    cond = df["rating"] < 0.5
    df["rating"].where(cond, 0, inplace=True)
    df["rating"].where(~cond, 1, inplace=True)

    enc_movie = {movie_id: idx for idx, movie_id in enumerate(df["movieId"].unique())}
    df["movieId"] = [enc_movie[movie_id] for movie_id in df["movieId"]]
    print(f"movieId is from {df['movieId'].min()} to {df['movieId'].max()}")

    enc_user = {user_id: idx for idx, user_id in enumerate(df["userId"].unique())}
    df["userId"] = [enc_user[user_id] for user_id in df["userId"]]
    print(f"userId is from {df['userId'].min()} to {df['userId'].max()}")

    # PyTorch dataset
    class MovieLensSmall(torch.utils.data.Dataset):
        def __init__(self, df: pd.DataFrame) -> None:
            self.df = df

        def __len__(self) -> int:
            return len(self.df)

        def __getitem__(self, idx: int):
            return list(df.iloc[idx])

    train_dataloader = torch.utils.data.DataLoader(
        MovieLensSmall(df),
        batch_size=4,
        shuffle=True,
        num_workers=8,
    )

    model = GMFModel(num_users=len(enc_user), num_items=len(enc_movie), embedding_dim=10)
    model = nn.DataParallel(model)
    model.train()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-3,
    )

    # Training loop
    log_idx = 1_000
    for epoch in range(10):
        running_loss = 0.0
        for idx, (users, items, ratings) in enumerate(train_dataloader):
            # move users, items, and ratings onto the device
            users = users.cuda().long()
            items = items.cuda().long()
            ratings = ratings.cuda()
            # users, items, ratings = users.cuda().long(), items.cuda.long(), ratings.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(users, items).reshape(-1)

            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()

            # accumulate loss and log
            running_loss += loss.item()
            if idx % log_idx == log_idx - 1:
                print(f"Epoch {epoch} | Steps: {idx + 1:<4} | Loss: {running_loss / log_idx:.3f}")
                running_loss = 0.0

# }}}
