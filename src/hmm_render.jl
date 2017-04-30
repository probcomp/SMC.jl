using PyPlot
using SMC

function render!(hmm::HiddenMarkovModel)
    # renders a HMM in the current figure
    # example usage:
    # plt[:figure]()
    # render!(hmm)
    # plt[:savefig]("hmm.png")
    num_states = hmm.num_states
    num_obs = hmm.num_obs
    # three subplots left to right:
    # 1. the prior (column vector)
    # 2. the transition matrix (square matrix)
    # 3. the observation matrix (rectangle)
    # the height is always num_states
    width_ratios = [1, num_states, num_obs]
    gs = matplotlib[:gridspec][:GridSpec](1, 3, width_ratios=width_ratios)
    prior_ax = plt[:subplot](gs[1])
    trans_ax = plt[:subplot](gs[2])
    obs_ax = plt[:subplot](gs[3])
    prior_mat = Array{Float64,2}(num_states,1)
    prior_mat[:,1] = hmm.initial_state_prior
    prior_ax[:imshow](prior_mat, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
    trans_ax[:imshow](hmm.transition_model, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
    obs_ax[:imshow](hmm.observation_model, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
    prior_ax[:set_title]("start model")
    trans_ax[:set_title]("transition model")
    obs_ax[:set_title]("observation model")
    for ax in [prior_ax, trans_ax, obs_ax]
        ax[:set_yticks](0:num_states-1)
        ax[:set_yticklabels](1:num_states)
    end
    prior_ax[:set_ylabel]("initial state")
    prior_ax[:set_xticks]([])
    trans_ax[:set_ylabel]("previous state")
    trans_ax[:set_xlabel]("next state")
    trans_ax[:set_xticks](0:num_states-1)
    trans_ax[:set_xticklabels](1:num_states)
    obs_ax[:set_ylabel]("state")
    obs_ax[:set_xlabel]("observation")
    obs_ax[:set_xticks](0:num_obs-1)
    obs_ax[:set_xticklabels](1:num_obs)
    plt[:tight_layout]()
end
