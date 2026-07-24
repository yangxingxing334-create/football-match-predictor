Swedish Allsvenskan season files in this folder were normalized for this project from:
https://github.com/bcalves/soccer_data/blob/2d000ceed38f79be6ca261a460035cc787a23294/Sweden_Allsvenskan.zip

Only the columns required by this repository's training pipeline are retained:
Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HTHG, HTAG, HTR, HS, AS, HST, AST, HR, AR.

The checked-in Swedish season files start at 2013 because the available 2010-2011 source rows did not include complete values for this repository's required training features, and the upstream archive did not include a 2012 season file.
