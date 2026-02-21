import {useState} from "react";
import {invoke} from "@tauri-apps/api/core";
import {Button, Game} from "./components";
import '../css/MainMenu.css';
import '../css/MenuDiv.css';
import '../css/Body.css';
import {ChessGame} from "./representation/ChessGame.ts";
import {Message} from "./representation/Message.ts";


const App = () => {
    // 1. Define state INSIDE the component
    // We store the actual game data, not the JSX
    const [activeGame, setActiveGame] = useState<ChessGame | null>(null);

    // 2. Define the logic inside so it can access 'setActiveGame'
    async function handleOpenGame() {
        const rawJson: string = await invoke('import_game', { gameType: 'classic' });
        const message: Message = JSON.parse(rawJson);

        if (message.type === 'error') {
            alert(message.message);
            return;
        }

        // Parse the actual game object from the message
        const gameData: ChessGame = JSON.parse(message.message);
        setActiveGame(gameData);
    }

    // 3. Define the Main Menu view
    const renderMainMenu = () => (
        <div className='menu-div'>
            <Button text='Play' selected />
            <Button text='Classic Chess' />
            <Button text='Analyze' onClick={handleOpenGame} />
            <Button text='Friends' />
            <Button text='Account' />
        </div>
    );

    // 4. Conditional Rendering (The "React Way")
    return (
        <main className='main-menu'>
            {activeGame ? (
                <Game
                    variant={activeGame.variant}
                    dimensions={activeGame.dimensions}
                    state={activeGame.state}
                />
            ) : (
                renderMainMenu()
            )}
        </main>
    );
};

export default App;