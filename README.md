# GoldenEye
<h1 align="center">
ML Deep-Q Network That Digs For Gold </h1>
<hr/>

<p>
I made this project to learn more about RNNs. The AI is given a randomly generated gold mine, which is a 2d array with integer values ranging from 0 - 270, organized into pockets of minerals using a combination of perlin and voronoi noise. <sub>(I based this off of minecraft)</sub>
</p>

<h2 align="center">
Mine Examples
</h2>

<p align="center">
<img src="https://user-images.githubusercontent.com/47166254/196258588-666f25c4-7ac3-469a-93cd-079989b135b7.png" width="320" />
<img src="https://user-images.githubusercontent.com/47166254/196259997-c5986c33-5258-434b-aefc-7e42086265e8.png" width="320" />
</p>

<p>
It's purpose is to reach a goal point and get the best score possible by collecting the stored values in its chosen path.

As the AI moves across each index of the array, it also loses n points. This is what I call the greed factor, it dictates how greedy it is allowed to be while attempting to reach a goal.
</br>


<h2 align="center">
First Model
</h2>

<p align="center">
<img src="https://github.com/Francis-Bui/GoldenEye/blob/main/png/training_time.png" width="500" title="Training Time"</p>

<p align="center">
<em> (Training Time) </em>
</p>

This initial version of the model was trained through 6000 episodes on a single map, with a high greed factor, a fixed start point of [0,0], and endpoint of [999,999]

Quite the obvious oversight was made I created this model. This model was able to traceback its steps and had too high of a greed factor for a [1000x1000] mine. This lead to a ridicoulously long training time as the model would essentially max out its movements, time out, and just go in circles. After it slowly begins to learn, it ended up finding two of the highest value pixels in the mine and would just bounce between the two to max out it's score.

I waited ten longs days for this model to finsih training, I saw fairly high scores and didn't notice the timouts so I was very excited to see what it could do. For a good moment it displayed correct path finding, but I noticed that it was not to the goal point. I watched in awe as the model ignored the goal entirely and brought its score as close as it could to infinity. 

Lot's of time spent, but lot's of things learned.

<h2 align="center">
Second Model
</h2>
<p align="center"><img src="https://github.com/Francis-Bui/GoldenEye/blob/main/png/Figure_2.png" width="500"/></p>

There were many solutions to fix the previous issues, but I landed on preventing the AI from being able to go back to where it previously was. I also shrunk the size of the mine to 100x100.

This way, training time improved drastically (27x faster)!

This model was trained through 6000 episodes on a single map, with a medium greed factor, a fixed start point of [0,0], and endpoint of [99,99].

Firstly, this AI must overcome the challenge of not trapping itself and being unable to move - like the game snake.

The AI had learned that if it made an equal number of positive x and positive y movements it would be unable to trap itself.

After that, the AI has learned to take an efficient path through the noise and reach the goal.

What it does to find the goal is travel through the best path until it reaches the wall, after it is no longer able to move in the positive x axis, it immediately travels in the positive y axis because it has learned that the goal will always be there.

Is this the most efficient path? Sort of... Score wise, it is not; however the AI has taken advantage of its ruleset and goalpoint, overfitting itself.

<h2 align="center">
Third Model
</h2>

I didn't want the model to just follow walls to get to it's point, I wanted to see how it would learn to find one. So instead of fixed spawn/goal points, I have randomized them.

This model is currently being trained for 30000 episodes on a single 50x50 mine. Still signle because I want the model to learn to naviagate this mine in particular.

