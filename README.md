# poem_bot
Poem generation with LSTM trained on a Haiku database. A deployed version can be found at http://poemmyface.herokuapp.com/index (it can be a bit slow to access if it has seen no traffic within the hour). That particular model did not learn the Haiku structure as it was trained on only 30 character blocks - hardware ram limitations :(. Therefore I am letting the poems be of somewhat arbitrary lengths. The model did pick up the short property of the haiku's verse.
