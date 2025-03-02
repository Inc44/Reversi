let gameState = {
	board: [],
	currentPlayer: 1,
	validMoves: [],
	gameOver: false,
	winner: 0,
	blackScore: 2,
	whiteScore: 2
};
const BLACK = 1;
const WHITE = -1;
const EMPTY = 0;
async function initGame()
{
	const boardSize = parseInt(document.getElementById('board-size')
		.value);
	const blackPlayer = document.getElementById('black-player')
		.value;
	const whitePlayer = document.getElementById('white-player')
		.value;
	try
	{
		const response = await fetch('/api/init',
		{
			method: 'POST',
			headers:
			{
				'Content-Type': 'application/json',
			},
			body: JSON.stringify(
			{
				boardSize,
				blackPlayer,
				whitePlayer
			}),
		});
		const result = await response.json();
		if (result.error)
		{
			showStatusMessage(result.error, 'error');
			return;
		}
		updateGameState(result);
		if (blackPlayer === "AI" && whitePlayer === "AI")
		{
			setTimeout(handleAIvsAI, 1000);
		}
	}
	catch (error)
	{
		showStatusMessage('Error initializing game: ' + error, 'error');
	}
}

function updateGameState(newState)
{
	gameState = newState;
	renderBoard();
	updateGameInfo();
	if (gameState.gameOver)
	{
		showGameOver();
	}
}

function renderBoard()
{
	const gameBoard = document.getElementById('game-board');
	gameBoard.innerHTML = '';
	const boardSize = gameState.board.length;
	gameBoard.style.gridTemplateColumns = `repeat(${boardSize}, 1fr)`;
	for (let row = 0; row < boardSize; row++)
	{
		for (let col = 0; col < boardSize; col++)
		{
			const cell = document.createElement('div');
			cell.className = 'cell';
			cell.dataset.row = row;
			cell.dataset.col = col;
			const isValidMove = gameState.validMoves.some(move => move[0] === row && move[1] === col);
			if (isValidMove)
			{
				cell.classList.add('valid-move');
			}
			if (gameState.board[row][col] !== EMPTY)
			{
				const piece = document.createElement('div');
				piece.className = `piece ${gameState.board[row][col] === BLACK ? 'black' : 'white'}`;
				cell.appendChild(piece);
			}
			cell.addEventListener('click', () => handleCellClick(row, col));
			gameBoard.appendChild(cell);
		}
	}
}
async function handleCellClick(row, col)
{
	if (gameState.gameOver) return;
	const isValidMove = gameState.validMoves.some(move => move[0] === row && move[1] === col);
	if (!isValidMove) return;
	try
	{
		const response = await fetch('/api/move',
		{
			method: 'POST',
			headers:
			{
				'Content-Type': 'application/json',
			},
			body: JSON.stringify(
			{
				row,
				col
			}),
		});
		const result = await response.json();
		if (result.error)
		{
			showStatusMessage(result.error, 'error');
			return;
		}
		updateGameState(result);
	}
	catch (error)
	{
		showStatusMessage('Error making move: ' + error, 'error');
	}
}
async function handleAIvsAI()
{
	if (gameState.gameOver) return;
	const blackPlayer = document.getElementById('black-player')
		.value;
	const whitePlayer = document.getElementById('white-player')
		.value;
	if (blackPlayer === "AI" && whitePlayer === "AI")
	{
		try
		{
			const response = await fetch('/api/ai-move',
			{
				method: 'POST',
				headers:
				{
					'Content-Type': 'application/json',
				},
			});
			const result = await response.json();
			if (result.error)
			{
				showStatusMessage(result.error, 'error');
				return;
			}
			updateGameState(result);
			if (!result.gameOver)
			{
				setTimeout(handleAIvsAI, 1000);
			}
			if (result.gameOver)
			{
				initGame();
			}
		}
		catch (error)
		{
			showStatusMessage('Error: ' + error, 'error');
		}
	}
}

function updateGameInfo()
{
	const currentPlayerEl = document.getElementById('current-player');
	currentPlayerEl.textContent = gameState.currentPlayer === BLACK ? 'Black' : 'White';
	document.getElementById('black-score')
		.textContent = gameState.blackScore;
	document.getElementById('white-score')
		.textContent = gameState.whiteScore;
}

function showGameOver()
{
	let message = '';
	if (gameState.winner === BLACK)
	{
		message = 'Black wins!';
	}
	else if (gameState.winner === WHITE)
	{
		message = 'White wins!';
	}
	else
	{
		message = "It's a draw!";
	}
	showStatusMessage(message, 'success');
}

function showStatusMessage(message, type)
{
	const statusEl = document.getElementById('status-message');
	statusEl.textContent = message;
	statusEl.className = type;
}
document.getElementById('new-game-btn')
	.addEventListener('click', initGame);
document.addEventListener('DOMContentLoaded', initGame);