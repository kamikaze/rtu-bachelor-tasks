resource "aws_ecr_repository" "backend" {
  count = var.ecr_enabled ? 1 : 0
  name  = "${var.project}-backend"

  #  lifecycle {
  #    prevent_destroy = true
  #  }

  lifecycle {
    ignore_changes = [
      name,
    ]
  }

  tags = {
    Name         = "${var.project}-backend"
    Organization = var.organization
    Project      = var.project
  }
}

resource "aws_ecr_repository" "frontend" {
  count = var.ecr_enabled ? 1 : 0
  name  = "${var.project}-frontend"

  #  lifecycle {
  #    prevent_destroy = true
  #  }

  lifecycle {
    ignore_changes = [
      name,
    ]
  }

  tags = {
    Name         = "${var.project}-frontend"
    Organization = var.organization
    Project      = var.project
  }
}
