import { ChessGame } from "../../representation/ChessGame.ts";
import { Tile } from "./Tile.tsx";
import {useEffect, useState} from "react";


const WhiteTile = '#a8a7a3';
const BlackTile = '#353223';


export const Game = ({variant, dimensions, state}: ChessGame) => {
    // Shorthand notation
    const [ columns, rows ] = dimensions;
    let [pieces, setPieces] = useState(structuredClone(state.starting_board));

    // Set the grid size dynamically
    const boardStyle = {
        display: 'grid',
        gridTemplateColumns: `repeat(${columns}, 60px)`,
        gridTemplateRows: `repeat(${rows}, 60px)`,
        border: '5px solid black'
    };

    let [moves, setMoves] = useState(-1);
    useEffect(() => {
        const handleKey = (event: KeyboardEvent) => {
            if ( event.key === 'ArrowLeft' || event.key === 'ArrowRight' ) {
                // Calculate the move index
                setMoves(Math.min(Math.max(0, moves + (event.key === 'ArrowRight' ? 1 : -1)), state.moves.length));

                let newPieces = structuredClone(state.starting_board);
                for ( let i = 0; i < moves; ++i ) {
                    // Move the piece
                    let from = state.moves[i].from[0] * columns + state.moves[i].from[1];
                    let to = state.moves[i].to[0] * columns + state.moves[i].to[1];
                    if (state.moves[i].promotion) newPieces[to] = {
                        piece_type: state.moves[i].promotion,
                        player: state.moves[i].player,
                        first_move: false
                    }
                    else newPieces[to] = newPieces[from];
                    newPieces[from] = null;
                }
                setPieces(newPieces);
            }
        }
        window.addEventListener('keydown', handleKey);
        return () => window.removeEventListener('keydown', handleKey);
    }, [moves]);


    // Generate the board
    return (
        <div style={boardStyle}>
            { pieces.map((piece, index) => {
                // Calculate X and Y
                const x = index % columns;
                const y = Math.floor(index / columns);
                const color = (x + y) % 2 === 0 ? WhiteTile : BlackTile;

                return (
                    <Tile color={color} piece={piece} key={index}></Tile>
                );
            })}
        </div>
    );
};