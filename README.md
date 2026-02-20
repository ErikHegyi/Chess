# Description
Chess has stayed pretty much the same for hundreds, if not thousands of years now. I saw a cool new variant online,
and I thought that this would be fun to play, but sadly, nothing like that actually existed.
So, I decided to create a chess engine and a UI for it, so that I, and in the future, hopefully others as well,
could play this cool new variant. From that point, anytime I had an idea for a new variant, I added it.

# How does the model work?
The core idea is pretty simple:
- generate all the legal moves
- use a machine-learning model to rate each of these moves
- select the move, which received the highest rating

# Model Design
For the model, I used a convolutional neural network. I have a detailed description of the mathematics that went into
engineering the model structure [here](docs/mathematics/Mathematics.pdf).

# Variants
## Normal Chess
I feel like this does not require any explanation. It is just the good old chess. If anyone is interested in the rules,
they can look up the rules to chess online.