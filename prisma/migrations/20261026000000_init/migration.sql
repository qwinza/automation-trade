-- CreateTable
CREATE TABLE `TradeDecision` (
    `id` VARCHAR(191) NOT NULL,
    `asset` VARCHAR(191) NOT NULL,
    `timeframe` VARCHAR(191) NOT NULL,
    `action` VARCHAR(191) NOT NULL,
    `confidence` INTEGER NOT NULL,
    `reasoning` TEXT NOT NULL,
    `market_context` JSON NOT NULL,
    `created_at` DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),

    PRIMARY KEY (`id`)
) DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
