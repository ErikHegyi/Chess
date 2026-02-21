interface Piece {
    piece_type: String,
    player: number,
    first_move: boolean
}


interface Move {
    player: number,
    from: [number , number],
    to: [number, number],
    promotion: String | null
}


interface State {
    starting_board: Array<Piece | null>,
    board: Array<Piece | null>,
    mask: Array<boolean>,
    player_to_move: number,
    number_of_players: number,
    moves: Array<Move>,
    last_capture: number
}

export interface ChessGame {
    variant: String,
    dimensions: Array<number>,
    state: State
}