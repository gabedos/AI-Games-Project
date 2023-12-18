Blackjack:
	echo "#!/bin/bash" > Blackjack
	echo "python3 game.py \"\$$@\"" >> Blackjack
	chmod u+x Blackjack
